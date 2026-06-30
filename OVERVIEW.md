# VietLex Nexus — Vietnamese SME Legal RAG

VietLex Nexus is a legal retrieval and question answering pipeline for the **Road to AI 2026 (R2AI) — Build AI Legal Assistant** challenge.
It is designed by parsing legal PDFs, retrieving the most relevant legal articles, and generating a submission file answering a set of 2000 questions with legal citations.

The output is a competition-style JSON/ZIP submission containing:

```json
{
  "id": 1,
  "question": "...",
  "answer": "...",
  "relevant_docs": ["<document_code>|<document_title>"],
  "relevant_articles": ["<document_code>|<document_title>|<Điều X>"]
}
```

---

## 1. Purpose

Vietnamese legal questions often require matching a user’s question to the correct legal document, article, clause, and sometimes a cited article in another law. This is difficult because:

- Vietnamese legal PDFs may contain noisy spacing and OCR-like extraction artifacts.
- Legal articles often cite other articles, clauses, decrees, laws, or circulars.
- A high lexical match can still point to the wrong legal family.
- Competition scoring rewards correct article retrieval, especially recall, not only fluent answers.

This notebook therefore focuses on:

- robust PDF legal-text parsing;
- article- and clause-aware chunking;
- hybrid retrieval using lexical, vector, and reranking signals;
- semantic legal-event routing;
- reference-graph assisted article selection;
- extractive-first answer generation;
- validated R2AI submission formatting.

The system is a legal-information assistant for initial reference, not a replacement for lawyers or official legal advice.

---

## 2. Notebook Pipeline Overview

The cleaned notebook is organized as a complete end-to-end RAG pipeline:

```text
Raw legal PDFs
   ↓
PDF text extraction and normalization
   ↓
Article parsing and validation
   ↓
Article-level filtering and text lock
   ↓
Legal-unit chunking
   ↓
SQLite storage and lookup tables
   ↓
Legal reference graph
   ↓
BM25 + Chroma vector retrieval
   ↓
Cross-encoder reranking
   ↓
Semantic question understanding
   ↓
Task-aware article selection
   ↓
Reference graph context enrichment
   ↓
Grounded answer generation
   ↓
Submission JSON/ZIP validation
```

---

## 3. Main Notebook Sections

### 1. Environment and Configuration

Defines runtime settings, paths, model names, feature flags, and output locations.

Important configuration groups include:

- input legal PDF folder;
- question JSON path;
- SQLite database path;
- Chroma index path;
- BM25/index cache paths;
- embedding and reranker model settings;
- optional local LLM flags;
- batch-run controls.

The notebook is designed so that expensive optional runs, such as sample tests or full submission generation, are disabled by default until explicitly enabled.

---

### 2. Imports and Utilities

Loads common Python libraries and utility helpers used throughout the pipeline.

This section centralizes:

- JSON and path handling;
- string normalization helpers;
- safe display utilities;
- compact debugging helpers;
- reusable text and metadata functions.

---

### 3. PDF Extraction, Text Normalization, and Article Parsing

This section extracts text from legal PDFs and converts each document into article-level records.

It includes:

- PDF text extraction using a fast text-first strategy;
- optional fallback extraction modes;
- Vietnamese legal text repair;
- spacing repair for glued Vietnamese syllables;
- abbreviation-aware normalization;
- article heading detection;
- guards against false article headings caused by inline citations;
- parser coverage audits.

A key design principle is:

```text
Inline references must stay inside the real parent article.
They must not become fake parent articles.
```

For example, when a sanction article says:

```text
... theo khoản 3 Điều 144 của Bộ luật Lao động ...
```

the parser should keep this text inside the current article and should not create a fake article such as:

```text
12/2022/NĐ-CP|Điều 144
```

---

### 4. Article Filtering and Text Lock

This section validates parsed article records before chunking.

It checks that each article has:

- a valid document code;
- a valid document title;
- a valid article number;
- usable legal text;
- sufficiently clean extracted content.

The section also creates the locked article text that downstream components use. After this point, the notebook should not rely on raw PDF text directly.

---

### 5. Chunking and Chunk Quality

This section splits valid article text into retrieval chunks.

The chunker keeps legal hierarchy metadata such as:

- document code;
- document title;
- article number;
- article title;
- clause number;
- point number;
- chunk label;
- parent article ID.

The goal is not to create arbitrary fixed-size chunks, but to preserve legal units whenever possible.

Chunk quality audits check for:

- suspicious parent-heading artifacts;
- extremely short or noisy chunks;
- glued text patterns;
- long-tail chunk counts;
- missing parent metadata.

---

### 6. Storage, Lookup Tables, and Reference Graph

This section builds persistent and in-memory legal data structures.

It creates:

- SQLite tables for documents, articles, and chunks;
- article lookup dictionaries;
- chunk lookup dictionaries;
- document/article metadata indexes;
- legal reference graph edges.

The reference graph captures detected citations such as:

```text
source article/chunk → cited article/clause/chunk
```

Example:

```text
12/2022/NĐ-CP Điều 29 Khoản 1
    cites → 45/2019/QH14 Điều 144 Khoản 3
```

This helps the system distinguish between:

- the article that owns a chunk; and
- the article merely referenced by that chunk.

---

### 7. Retrieval Indexes

This section builds or loads retrieval indexes.

The notebook uses:

- BM25 for lexical matching;
- dense embeddings for semantic retrieval;
- Chroma as the vector store;
- cached indexes where possible.

The retriever uses cleaned searchable text, while answer generation can still show the legal text in a more natural display form.

---

### 8. Retrieval and Reranking

This section performs hybrid retrieval and reranking.

The retriever combines:

- exact question text;
- normalized question text;
- salient legal phrases;
- event-specific query expansions;
- role-specific legal query expansions;
- BM25 results;
- vector results;
- cross-encoder reranker scores;
- semantic priors.

The output is a ranked set of candidate chunks with metadata.

---

### 9. Semantic Understanding and Document Access Control

This section converts a natural-language question into a legal frame.

The frame may include:

- domain;
- task type;
- legal event;
- active legal families;
- required roles;
- conditional roles;
- suppressed roles;
- negative domains/families;
- preferred legal references;
- event-specific retrieval queries.

Typical task types include:

- yes/no rule;
- deadline or amount;
- dossier/procedure;
- support-policy catalog;
- specific sanction amount;
- sanction remedy;
- authority question;
- multi-event question.

The document access policy also controls broad or noisy legal documents so they are only used when the question domain supports them.

---

### 10. Article Selection and Legal-Event Resolution

This section aggregates retrieved chunks into article candidates and selects the final legal articles.

Selection is not based only on chunk score. It also considers:

- question domain;
- legal event;
- article family;
- task role;
- exact phrase overlap;
- parent-article rescue;
- preferred references;
- cross-reference leakage guards;
- role coverage;
- reference graph expansion;
- event-specific veto rules.

The system is designed to avoid selecting unrelated articles that only match generic terms such as:

```text
phạt tiền từ
trong thời hạn
kể từ ngày
biện pháp khắc phục hậu quả
```

---

### 11. Answer Generation and Context Enrichment

This section builds the legal context and generates the final answer.

The notebook uses an extractive-first strategy:

1. retrieve and select legal articles;
2. extract relevant legal facts from selected articles;
3. build a grounded answer;
4. optionally use an LLM to improve wording;
5. clean and validate the final answer.

Reference graph context can add cited articles when they are legally necessary and pass selection guards.

---

### 12. Submission Formatting and Validation

This section formats each answer into the required R2AI submission object.

It validates:

- required keys;
- answer type;
- document reference format;
- article reference format;
- duplicate references;
- missing question IDs;
- JSON serializability.

The expected output item is:

```json
{
  "id": 1,
  "question": "...",
  "answer": "...",
  "relevant_docs": ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
  "relevant_articles": [
    "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 11"
  ]
}
```

---

### 13. Questions, Coverage Audit, and Regression Tests

This section loads the test questions and provides diagnostic tools.

It includes:

- question loading;
- frame coverage audit;
- selected regression questions;
- debug reports for individual questions;
- checks for known failure modes.

Useful debugging commands include:

```python
debug_question("...")
```

and:

```python
audit_first_n_question_frames(500)
```

---

### 14. Submission Runs and Index Backup

This section runs the answer generation loop and saves outputs.

Common outputs include:

```text
results_first500.json
submission_first500.zip
results.json
submission.zip
```

The section also includes optional index backup utilities so expensive retrieval indexes can be reused across sessions.

---

## 4. Input Data

The notebook expects two main inputs.

### Legal PDFs

A folder containing Vietnamese legal PDF files, for example:

```text
raw_laws/
```

Each PDF should be a machine-readable legal document. The parser can handle many spacing artifacts, but completely scanned image-only PDFs may require OCR outside this notebook.

### Question JSON

A JSON file containing a list of questions:

```json
[
  {
    "id": 1,
    "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?"
  }
]
```

---

## 5. Output Format

The final submission is a list of objects:

```json
[
  {
    "id": 1,
    "question": "Các cơ sở ươm tạo và khu làm việc chung được hưởng những chính sách hỗ trợ nào về thuế và đất đai?",
    "answer": "Căn cứ pháp lý:\n...\n\nTrả lời:\n...",
    "relevant_docs": ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
    "relevant_articles": [
      "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 12"
    ]
  }
]
```

The generated ZIP should contain a flat `results.json` file unless the competition instructions specify another name.

---

## 6. Retrieval and Selection Strategy

The notebook is optimized for legal article retrieval rather than free-form chat.

The selection strategy follows these principles:

1. **Prefer exact legal-event evidence.**  
   If a top chunk directly answers the question, select its parent article before generic role-filling.

2. **Avoid fake parent articles.**  
   Inline references are represented as graph edges, not as article rows belonging to the current document.

3. **Use role coverage conservatively.**  
   A missing generic role should not pull an unrelated article from another legal family.

4. **Separate similar legal families.**  
   For example, SME production land-rent support is different from incubator/co-working-space support.

5. **Preserve article-level submission references.**  
   Even if a chunk is clause-level, the competition output should cite the parent legal article.

---

## 7. Optional LLM Usage

The notebook can optionally use a local or hosted LLM for:

- frame review;
- candidate verification;
- answer wording.

The default recommendation is:

```python
USE_LLM_FRAME_CLASSIFIER = False
USE_LLM_CANDIDATE_VERIFIER = False
```

The rule-based semantic layer should be tested first. Candidate verification can be enabled later for difficult cases, but it should ideally run only when the rule-based selector is uncertain.

---

## 8. Running the Notebook

A typical full run is:

```text
1. Environment and Configuration
2. Imports and Utilities
3. PDF Extraction, Text Normalization, and Article Parsing
4. Article Filtering and Text Lock
5. Chunking and Chunk Quality
6. Storage, Lookup Tables, and Reference Graph
7. Retrieval Indexes
8. Retrieval and Reranking
9. Semantic Understanding and Document Access Control
10. Article Selection and Legal-Event Resolution
11. Answer Generation and Context Enrichment
12. Submission Formatting and Validation
13. Questions, Coverage Audit, and Regression Tests
14. Submission Runs and Index Backup
```

When parser, chunking, or text-cleaning logic changes, rebuild from section 3 onward and recreate SQLite, BM25, and Chroma indexes.

When only semantic selection logic changes, existing chunks and indexes can usually be reused.

---

## 9. Evaluation Focus

The R2AI task evaluates both retrieval and answer quality.

### Retrieval

The system should return correct legal documents and articles. The challenge emphasizes recall, so missing a required article can be more damaging than including a small number of extra relevant articles.

### Answer Quality

The generated answer should be:

- legally grounded;
- faithful to selected articles;
- complete enough for the question;
- practical for SME users;
- clear and non-hallucinatory.

---

## 10. Known Limitations

The notebook still has practical limitations:

- Some legal PDFs may contain extraction artifacts that require manual inspection.
- Some questions require multiple legal events and may need broader article coverage.
- Reference resolution is heuristic and may not resolve every cited article.
- The semantic event registry may not cover every possible legal family.
- LLM-based answer generation may still need output validation.
- This notebook is for competition and research use, not official legal advice.

---

## 11. Future Improvements

Possible future improvements include:

- stronger Vietnamese legal abbreviation expansion;
- better clause-to-clause reference resolution;
- more legal-event families;
- learned candidate verification;
- better multi-event question decomposition;
- automatic parser coverage reports per legal document;
- active-learning loops from failed debug questions;
- separate production API or user interface.

---

## 12. Disclaimer

This project is for educational, research, and competition purposes.  
Generated answers are for reference only and should not be considered official legal advice. Important legal decisions should be verified with official legal documents and qualified legal professionals.
