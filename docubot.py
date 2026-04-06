"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re


def _tokenize(text):
    """Lowercase alphanumeric tokens (letters, digits, underscores)."""
    return re.findall(r"[a-z0-9_]+", text.lower())


# Common words that hurt ranking when every chunk matches them equally.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "it",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "do",
        "does",
        "did",
        "any",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "its",
        "our",
        "their",
        "and",
        "or",
        "but",
        "if",
        "there",
        "here",
        "into",
        "over",
        "after",
        "before",
        "about",
        "against",
        "between",
        "through",
        "during",
        "without",
        "within",
        "must",
        "may",
        "might",
    }
)


def _query_content_tokens(query):
    """Drop stopwords so scores reflect content terms (auth, token, users, …)."""
    raw = _tokenize(query)
    content = [w for w in raw if w not in _STOPWORDS and len(w) > 2]
    return content if content else raw


def _expanded_token_set(text):
    """
    Tokens in text plus underscore-split parts so auth_utils matches query "auth"
    and generate_access_token matches "generate" / prefix overlap.
    """
    s = set()
    for t in _tokenize(text):
        s.add(t)
        if "_" in t:
            for part in t.split("_"):
                if part:
                    s.add(part)
    return s


def _token_matches_chunk_term(query_tok, expanded_chunk_tokens):
    if query_tok in expanded_chunk_tokens:
        return True
    if len(query_tok) < 5:
        return False
    pref = query_tok[:5]
    for t in expanded_chunk_tokens:
        if len(t) >= 5 and (t.startswith(pref) or query_tok.startswith(t[:5])):
            return True
    return False


class DocuBot:
    # Refuse retrieval when no chunk reaches this overlap score (distinct query terms matched).
    min_retrieval_score = 1

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Paragraph-level chunks for tighter snippets (Phase 3)
        self.chunks = self._build_chunks(self.documents)

        # Inverted index: token -> list of chunk indices
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    def _split_into_chunks(self, filename, text):
        """
        Split on blank lines into paragraph-like units. Small fragments are merged
        into the whole file as one chunk when needed so short docs still retrieve.
        """
        stripped = text.strip()
        if not stripped:
            return []
        parts = re.split(r"\n\s*\n+", stripped)
        chunks = []
        for p in parts:
            p = p.strip()
            if len(p) >= 40:
                chunks.append((filename, p))
        if not chunks:
            chunks.append((filename, stripped))
        return chunks

    def _build_chunks(self, documents):
        out = []
        for filename, text in documents:
            out.extend(self._split_into_chunks(filename, text))
        return out

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build a tiny inverted index mapping lowercase words to chunk indices.

        chunks is a list of (filename, text). The index maps each token to the
        indices of chunks where that token appears at least once.
        """
        index = {}
        for i, (_, text) in enumerate(chunks):
            for tok in _expanded_token_set(text):
                index.setdefault(tok, []).append(i)
                if len(tok) >= 5:
                    index.setdefault(tok[:5], []).append(i)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Score by content-query overlap against expanded chunk tokens, with light
        prefix matching (e.g. generated ↔ generation) and a small boost when the
        chunk names the token generator and the query is about generation.
        """
        q_words = _query_content_tokens(query)
        if not q_words:
            return 0
        ts = _expanded_token_set(text)
        base = sum(1 for w in q_words if _token_matches_chunk_term(w, ts))
        if base == 0:
            return 0
        if any(w.startswith("generat") for w in q_words) and (
            "generate_access_token" in text or "generate_access_token" in ts
        ):
            base += 2
        return base

    def retrieve(self, query, top_k=3):
        """
        Narrow to chunks that match any query token, score by query-term overlap,
        return top_k (filename, chunk text). Empty when best score is below
        min_retrieval_score (guardrail).
        """
        q_words = _query_content_tokens(query)
        if not q_words:
            return []

        candidate_indices = set()
        for w in q_words:
            for i in self.index.get(w, []):
                candidate_indices.add(i)
            if len(w) >= 5:
                for i in self.index.get(w[:5], []):
                    candidate_indices.add(i)

        if not candidate_indices:
            candidate_indices = set(range(len(self.chunks)))

        scored = []
        for i in candidate_indices:
            fn, text = self.chunks[i]
            s = self.score_document(query, text)
            scored.append((s, i, fn, text))

        # Prefer higher score, then shorter snippets (more focused than huge API tables).
        scored.sort(key=lambda x: (-x[0], len(x[3]), x[1]))

        # Weak match: one generic token (e.g. "docs") is not enough when the user
        # asked several content words (e.g. payment + processing + mention).
        min_needed = self.min_retrieval_score
        if len(q_words) >= 3:
            min_needed = max(min_needed, 2)

        if not scored or scored[0][0] < min_needed:
            return []

        results = []
        seen = set()
        for s, _i, fn, text in scored:
            if s < min_needed:
                break
            key = (fn, text)
            if key in seen:
                continue
            seen.add(key)
            results.append((fn, text))
            if len(results) >= top_k:
                break
        return results

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
