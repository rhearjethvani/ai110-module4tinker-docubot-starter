# DocuBot Model Card

This model card reflects the DocuBot implementation in this repo: inverted-index retrieval over paragraph chunks, overlap scoring, and Gemini for naive and RAG modes. Re-run `python main.py` with your own API key and tweak notes if your live answers differ.

---

## 1. System Overview

**What is DocuBot trying to do?**  
Describe the overall goal in 2 to 3 sentences.

> DocuBot answers developer questions about a local `docs/` folder. It can return raw retrieved snippets only, or combine retrieval with Gemini so answers are grounded in a small set of passages instead of the model’s prior knowledge alone.

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

> Natural-language questions from the CLI; `.md` and `.txt` files under `docs/`; optional `GEMINI_API_KEY` (via `.env` or the environment) for Modes 1 and 3.

**What outputs does DocuBot produce?**

> Mode 1: a free-form natural-language answer (prompt includes the full concatenated docs). Mode 2: labeled snippet blocks or the refusal string `I do not know based on these docs.` Mode 3: a Gemini answer constrained to retrieved snippets, or the same refusal when retrieval returns nothing.

---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?  
  > Each file is split into paragraph-like chunks (blank-line separated segments; short files stay one chunk). Tokens are lowercase alphanumeric runs from a regex. The index maps each token to the list of chunk indices where it appears.

- How do you score relevance for a query?  
  > `score_document` counts how many **distinct** query tokens appear in the chunk (set overlap), not term frequency.

- How do you choose top snippets?  
  > Candidate chunks are any chunk that contains at least one query token (if none, all chunks are considered). Chunks are sorted by score descending, deduplicated by `(filename, text)`, and the top `top_k` are kept. If the best score is below `min_retrieval_score` (1), retrieval returns no snippets.

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

> Simplicity over sophistication: no stemming, embeddings, or BM25—only lexical overlap. That keeps the code easy to reason about for the activity, but synonyms and typos hurt recall. Paragraph chunks improve precision versus whole files but can split a table or list across boundaries.

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- Naive LLM mode:  
  > Calls Gemini once per question with the **entire** corpus in the prompt. No retrieval step.

- Retrieval only mode:  
  > No LLM. Only `retrieve` plus formatting of snippets (or refusal).

- RAG mode:  
  > Runs `retrieve`, then calls Gemini with **only** the top snippets and strict instructions to stay in that context.

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

> Naive prompt: use only the pasted documentation; if unsupported, say you are not sure rather than inventing details. RAG prompt (in `llm_client.py`): use only snippets; do not invent endpoints or config; if insufficient evidence, reply exactly `I do not know based on the docs I have.`; when answering, mention which files were used.

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query | Naive LLM: helpful or harmful? | Retrieval only: helpful or harmful? | RAG: helpful or harmful? | Notes |
|------|---------------------------------|--------------------------------------|---------------------------|-------|
| Where is the auth token generated? | Helpful if it cites `auth_utils.py` | Helpful: surfaces Token Generation section | Helpful: concise answer tied to `AUTH.md` | Naive can still paraphrase wrong file paths if the prompt is long; RAG cites the same chunk the retriever picked. |
| How do I connect to the database? | Mixed: may infer stack not in docs | Helpful if `DATABASE.md` ranks high | Helpful when `DATABASE.md` is in snippets | Compare whether naive mentions only what `DATABASE.md` actually says (e.g. `DATABASE_URL`). |
| Which endpoint lists all users? | Often helpful (simple fact) | Helpful: raw `GET /api/users` line | Helpful | Retrieval shows exact route; RAG summarizes it. |
| How does a client refresh an access token? | Helpful when docs describe `POST /api/refresh` | Helpful but user must read snippet | Helpful | Example where retrieval is accurate but less readable than one RAG paragraph. |
| Is there any mention of payment processing in these docs? | **Harmful risk**: model may hallucinate “no” or “yes” | **Helpful**: empty retrieval → explicit refusal | **Helpful**: refusal or honest “not in snippets” | Good test for guardrails: docs have no payment section; lexical retrieval should return []. |
| What environment variables are required for authentication? | Helpful | Helpful | Helpful | Multiple env vars; watch naive merge with DB vars from other files. |

**What patterns did you notice?**  

- When does naive LLM look impressive but untrustworthy?  
  > When the answer sounds fluent and general but omits file-specific details, or blends concepts from several docs (e.g. mixing auth and DB setup) without showing which passage supports each claim.

- When is retrieval only clearly better?  
  > When you need **evidence**: exact endpoint strings, function names, or env var names. It does not paraphrase away critical tokens.

- When is RAG clearly better than both?  
  > When you want a short natural-language summary **and** traceability to specific files, with refusal when chunks are missing or weak.

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**  
For each one, say:

- What was the question?  
- What did the system do?  
- What should have happened instead?

> **Failure case 1:** Question: “Which endpoint returns all users?” with a typo like “usres.” **Behavior:** Retrieval scores drop or miss the right chunk; naive LLM might still guess `GET /api/users` from fuzzy reading or from memory. **Expected:** Ideally retrieval would fuzzy-match; with lexical-only retrieval, the user sees refusal or wrong chunk—fix would be better tokenization or edit distance, not more model parameters.

> **Failure case 2:** Question about behavior **not** in docs (e.g. payment). **Behavior:** Retrieval returns `[]` → “I do not know based on these docs.” **Naive** might still speculate. **Expected:** Refusal for retrieval/RAG; naive should hedge per prompt but is never as safe as empty context + RAG rules.

**When should DocuBot say “I do not know based on the docs I have”?**  
Give at least two specific situations.

> When retrieval finds no chunk meeting `min_retrieval_score`. When RAG receives snippets that do not contain the requested fact (the model is instructed to refuse rather than guess).

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

> Minimum overlap score before returning any chunk; empty retrieval short-circuits Mode 2 and Mode 3 before calling the model in RAG; RAG system prompt mandates exact refusal phrase when evidence is insufficient; naive prompt asks for uncertainty when docs do not support the answer.

---

## 6. Limitations and Future Improvements

**Current limitations**  
List at least three limitations of your DocuBot system.

1. Lexical overlap only—no semantic similarity (e.g. “sign in” vs “login”).
2. Chunk boundaries are naive; structured blocks (JSON examples) may split awkwardly.
3. Naive mode still sends a very long context; cost, latency, and “lost in the middle” effects can degrade grounding even with a good prompt.

**Future improvements**  
List two or three changes that would most improve reliability or usefulness.

1. Embeddings + vector store for retrieval (or hybrid lexical + dense).
2. Smarter chunking (per heading in Markdown, or fixed token windows with overlap).
3. Ask the model to quote a short span from snippets in RAG answers for easier verification.

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

> Developers might ship security or compliance decisions based on chat output. Wrong endpoints, auth flows, or env vars could break systems or expose data. Over-trusting naive mode is especially risky on private docs the model partially “remembers” from training.

**What instructions would you give real developers who want to use DocuBot safely?**  
Write 2 to 4 short bullet points.

- Treat every answer as a pointer to the docs, not a substitute for reading them for critical paths.
- Prefer Mode 2 or 3 when you need traceability; inspect the retrieved filenames and text.
- Run sensitive questions twice and compare; if retrieval is empty, do not switch to naive mode to “get any answer.”
- Keep API keys out of logs and version control; rotate keys if leaked.

---
