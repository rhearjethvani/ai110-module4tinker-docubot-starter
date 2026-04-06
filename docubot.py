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
            for tok in set(_tokenize(text)):
                index.setdefault(tok, []).append(i)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Simple relevance: count how many distinct query tokens appear in the text.
        """
        q_words = _tokenize(query)
        if not q_words:
            return 0
        text_tokens = set(_tokenize(text))
        return sum(1 for w in q_words if w in text_tokens)

    def retrieve(self, query, top_k=3):
        """
        Narrow to chunks that match any query token, score by query-term overlap,
        return top_k (filename, chunk text). Empty when best score is below
        min_retrieval_score (guardrail).
        """
        q_words = _tokenize(query)
        if not q_words:
            return []

        candidate_indices = set()
        for w in q_words:
            for i in self.index.get(w, []):
                candidate_indices.add(i)

        if not candidate_indices:
            candidate_indices = set(range(len(self.chunks)))

        scored = []
        for i in candidate_indices:
            fn, text = self.chunks[i]
            s = self.score_document(query, text)
            scored.append((s, i, fn, text))

        scored.sort(key=lambda x: (-x[0], x[1]))

        if not scored or scored[0][0] < self.min_retrieval_score:
            return []

        results = []
        seen = set()
        for s, _i, fn, text in scored:
            if s < self.min_retrieval_score:
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
