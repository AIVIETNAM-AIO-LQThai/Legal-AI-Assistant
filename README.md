# VietLex - instructions on reproducibility and requirements

## Project

**Project name:** SME Legal AI Assistant / RAGuru  
**Competition:** Road to AI 2026 - Build AI Legal Assistant  
**Main notebook:** `RAGuru(9).ipynb`  
**Recommended runtime:** Kaggle Notebook with **GPU T4 x2**  

This project implements a reproducible Vietnamese legal Retrieval-Augmented Generation (RAG) pipeline for SME-related legal questions. The notebook processes Vietnamese legal PDFs, extracts article-level legal records, creates clause/point-aware retrieval chunks, builds SQLite/BM25/Chroma indexes, retrieves and reranks legal evidence, selects relevant legal articles, generates Vietnamese answers, and exports the final `results.json` / `submission.zip` required by the competition.

The system is notebook-first. It is designed to be run end-to-end on Kaggle rather than as a deployed web application.

---

## 1. Data Documentation

### 1.1 Data Sources

The notebook uses the following data sources:

| Data type | Notebook path / expected location | Purpose |
| --- | --- | --- |
| Legal PDF corpus | `Your pdf path` | Source corpus for legal retrieval and answer grounding |
| Competition questions | `Your question path` | Input questions containing `id` and `question` |
| Prompt files | `Your prompts path` | Prompt templates for answer generation, validation, and SME counselor style |
| Working artifacts | `/kaggle/working/r2ai_pdf_legal_rag` | Generated SQLite database, BM25 index, Chroma index, caches, and intermediate outputs |
| Final outputs | `/kaggle/working/results.json` and `/kaggle/working/submission.zip` | Competition submission files |

Find Config in the notebook to set up the correct paths and other options like models to use, validator, answer, and counselor mode prompt activation.

The legal documents are Vietnamese legal texts stored as PDF files. The notebook extracts and normalizes their text, then converts them into structured legal records.

### 1.2 Input Data Format

The competition question file must be a JSON list of objects:

```json
[
  {
    "id": 1,
    "question": "Legal question in Vietnamese"
  }
]
```

Rules:

- `id` must be preserved exactly in the output.
- `question` must be preserved exactly and must not be rewritten.
- Each item is processed independently by the retrieval, selection, and answer-generation pipeline.

### 1.3 Processed Data Structure

The notebook converts raw PDFs into the following retrieval structure:

```text
Legal PDF
  -> extracted and normalized text
  -> document metadata
  -> article-level records
  -> clause/point-aware chunks
  -> SQLite storage
  -> BM25 keyword index
  -> Chroma vector index
  -> legal reference graph
  -> selected evidence articles
  -> generated answer
  -> results.json item
```

Main generated artifacts:

```text
/kaggle/working/r2ai_pdf_legal_rag/legal_pdf_store.sqlite
/kaggle/working/r2ai_pdf_legal_rag/bm25_pdf_index.pkl
/kaggle/working/r2ai_pdf_legal_rag/chroma_pdf/
/kaggle/working/results.json
/kaggle/working/submission.zip
```

If prebuilt artifacts are provided, `legal_pdf_store.sqlite` and `bm25_pdf_index.pkl` can be copied into `/kaggle/working/r2ai_pdf_legal_rag` and the rebuild flags can be turned off, as long as the artifacts were created from the same notebook/parser version.

### 1.4 Data Access

The data package should be shared through Google Drive, OneDrive, Kaggle Dataset, or an equivalent platform.

```text
Data link: https://drive.google.com/drive/folders/1qKJpbJHnrrfDwWSQ-xAru6CRpgez3khL?usp=sharing
```

The shared data package should include:

```text
raw_laws.zip or raw_laws/          # Legal PDF corpus
R2AIStage1DATA.json                # Competition input questions
prompts.zip or prompts/            # Prompt markdown files
```

Optional reusable artifacts may also be included:

```text
legal_pdf_store.sqlite             # Parsed articles/chunks in SQLite
bm25_pdf_index.pkl                 # BM25 index
chroma_pdf/                        # Vector index folder, if reused
```

When running on Kaggle, attach the data as a Kaggle input dataset and make sure the paths in Section 2 of the notebook match the mounted dataset paths.

---

## 2. Model and Checkpoint Documentation

### 2.1 Model Components

The notebook uses a hybrid legal RAG pipeline rather than relying only on a generative model.

| Component | Model / implementation | Purpose |
| --- | --- | --- |
| BM25 retriever | `rank_bm25` | Lexical keyword retrieval over normalized chunk text |
| Embedding model | `AITeamVN/Vietnamese_Embedding` | Dense semantic retrieval with Chroma |
| Vector store | ChromaDB | Persistent vector index for legal chunks |
| Reranker | `BAAI/bge-reranker-v2-m3` | Neural reranking of candidate chunks |
| Local LLM | `Qwen/Qwen2.5-3B-Instruct` | Optional prompted answer generation / answer rewriting |
| Rule-based semantic layer | Notebook code | Domain routing, legal-event detection, article selection, and role coverage |

The final notebook uses open-source models loaded by model ID. No custom fine-tuned model is trained inside the notebook.

### 2.2 Competition Model Restrictions

For the final reproducible submission, all model components satisfy the competition rules:

- The model weights must be publicly available.
- The model must be under 14B parameters.
- The model must have been released before **01/03/2026 Vietnam time**.
- The team must provide the model name, source, version or model ID, and usage instructions.
- Closed models such as GPT-4o, Gemini, or other non-compliant closed models must not be used for the final reproducible run.

The notebook is configured to use public Hugging Face model IDs. Before final submission, the team should record the exact model cards/source links and confirm that each model satisfies the release-date and size requirements.

### 2.3 Checkpoint List

Since this notebook only loads open-source models and does not fine-tune them, there is no custom trained checkpoint produced by the team. The checkpoint list is therefore the list of public model IDs used by the notebook:

| Component | Checkpoint / model ID | Purpose | How it is loaded |
| --- | --- | --- | --- |
| Embedding model | `AITeamVN/Vietnamese_Embedding` | Dense retrieval embeddings | Loaded by `sentence-transformers` / Hugging Face |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking | Loaded by `transformers` / Hugging Face |
| Local LLM | `Qwen/Qwen2.5-3B-Instruct` | Vietnamese answer generation / optional reasoning support | Loaded by `transformers`; 4-bit loading may be enabled |

If a Kaggle runtime has internet enabled, these models can be downloaded automatically from their public model hubs. If internet is disabled, the model folders must be downloaded in advance and attached as a Kaggle input dataset, then the notebook model paths should be updated accordingly.

### 2.4 Checkpoint Access

For reproduction, provide either:

1. Hugging Face model IDs and internet-enabled Kaggle execution, or  
2. A shared folder containing the downloaded model folders.

Suggested documentation format:

```text
Embedding model: AITeamVN/Vietnamese_Embedding
Reranker model: BAAI/bge-reranker-v2-m3
Local LLM: Qwen/Qwen2.5-3B-Instruct
Checkpoint access method: public Hugging Face download during Kaggle execution
Alternative offline checkpoint link: [insert Google Drive / Kaggle Dataset link if provided]
```

If using offline model folders, use a structure such as:

```text
checkpoints/
  Vietnamese_Embedding/
  bge-reranker-v2-m3/
  Qwen2.5-3B-Instruct/
```

Then update these notebook variables:

```python
EMBEDDING_MODEL_NAME = "path/to/Vietnamese_Embedding"
RERANKER_MODEL_NAME = "path/to/bge-reranker-v2-m3"
LOCAL_LLM_MODEL_NAME = "path/to/Qwen2.5-3B-Instruct"
```

---

## 3. Source Code Documentation

### 3.1 Main Source Code

The complete source code is contained in:

```text
VietLex.ipynb
```

The notebook pipeline is organized into these main stages:

```text
1. Environment setup
2. Configuration
3. Imports and shared utilities
4. PDF text extraction and normalization
5. Article filtering and domain metadata
6. Legal chunking strategy
7. Persistent storage and legal reference graph
8. BM25 and Chroma retrieval indexes
9. Hybrid retrieval and neural reranking
10. Semantic legal-event routing
11. Task-aware article selection
12. Extractive-first answer generation
13. Submission formatting and validation
14. Question loading, checks, submission generation, and index backup
```

The source code includes both retrieval infrastructure and legal-specific logic, including article-heading parsing, Vietnamese PDF text normalization, reference extraction, legal-event routing, domain gating, relevant article selection, and competition-format output generation.

### 3.2 Dependencies

The notebook installs or uses the following main Python packages:

```text
python>=3.10
pandas
numpy
regex
pymupdf>=1.24.0
rank_bm25
chromadb>=0.5.0
sentence-transformers>=3.0.0
transformers>=4.44.0
accelerate
sentencepiece
torch
tqdm
pyarrow
bitsandbytes              # needed when USE_4BIT_LLM=True
sqlite3                   # Python standard library
pickle                    # Python standard library
```

On Kaggle, the dependency installation cell can be kept enabled for a fresh runtime:

```python
INSTALL_DEPS = True
INSTALL_BITSANDBYTES = True
```

If all packages are already installed in the environment, these flags may be set to `False`.

### 3.3 Configuration Files

The notebook is mostly self-contained. The important configuration values are defined in Section 2 of `RAGuru(9).ipynb`.

Important paths:

```python
PDF_LAW_DIR = Path("/kaggle/input/datasets/thaile2007/legal-laws/raw_laws")
QUESTIONS_PATH = Path("/kaggle/input/datasets/thaile2007/questions/R2AIStage1DATA.json")
PROMPT_DIR = Path("/kaggle/input/datasets/thaile2007/prompts/prompts")
WORK_DIR = Path("/kaggle/working/r2ai_pdf_legal_rag")
DB_PATH = WORK_DIR / "legal_pdf_store.sqlite"
BM25_PATH = WORK_DIR / "bm25_pdf_index.pkl"
CHROMA_DIR = WORK_DIR / "chroma_pdf"
RESULTS_PATH = Path("/kaggle/working/results.json")
ZIP_PATH = Path("/kaggle/working/submission.zip")
```

Important model settings:

```python
EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LOCAL_LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
```

Important rebuild flags:

```python
REBUILD_SQLITE = True
REBUILD_BM25 = True
REBUILD_CHROMA = True
```

If cached artifacts are already available and match the same notebook version, these may be set to `False` to skip rebuilding:

```python
REBUILD_SQLITE = False
REBUILD_BM25 = False
REBUILD_CHROMA = False  # only if a compatible chroma_pdf folder is also available
```

---

## 4. Reproduction Guide

### 4.1 Environment Requirements

Recommended environment:

```text
Platform: Kaggle Notebook
Accelerator: GPU T4 x2
Python: >= 3.10
Internet: enabled if models are downloaded from Hugging Face at runtime
Disk: enough space for SQLite, BM25, Chroma, and model cache
RAM/GPU memory: T4 x2 recommended for embedding, reranking, and local LLM generation
```

The notebook includes low-VRAM settings for Kaggle T4-class GPUs, including optional 4-bit LLM loading and unloading models between retrieval/generation stages.

### 4.2 Setup Steps

Recommended Kaggle setup:

```text
1. Upload or attach the notebook `RAGuru(9).ipynb` to Kaggle.
2. Set the Kaggle accelerator to GPU T4 x2.
3. Enable internet if models will be downloaded from Hugging Face.
4. Attach the dataset containing legal PDFs, questions, and prompt files.
5. Verify that the notebook paths in Section 2 match the Kaggle input paths.
6. Run the notebook from top to bottom.
```

For a fresh Kaggle runtime, keep the installation flags enabled:

```python
INSTALL_DEPS = True
INSTALL_BITSANDBYTES = True
```

If using prebuilt SQLite/BM25 artifacts, copy them into the working directory before running the retrieval sections:

```python
from pathlib import Path
import shutil

src_dir = Path("/kaggle/input/datasets/thaile2007/prep-data")
dst_dir = Path("/kaggle/working/r2ai_pdf_legal_rag")
dst_dir.mkdir(parents=True, exist_ok=True)

for filename in ["bm25_pdf_index.pkl", "legal_pdf_store.sqlite"]:
    shutil.copy2(src_dir / filename, dst_dir / filename)
```

Then set the compatible rebuild flags to `False`.

### 4.3 Data and Checkpoint Setup

Data setup:

```text
1. Place legal PDFs under the configured `PDF_LAW_DIR`.
2. Place the competition question JSON at `QUESTIONS_PATH`.
3. Place prompt markdown files under `PROMPT_DIR`.
4. Confirm the paths printed by the configuration cell exist.
```

Model/checkpoint setup:

```text
1. If Kaggle internet is enabled, the notebook downloads the public models by model ID.
2. If Kaggle internet is disabled, download the model folders in advance and attach them as a Kaggle input dataset.
3. If using offline model folders, replace the Hugging Face model IDs in Section 2 with local folder paths.
```

There is no custom fine-tuned checkpoint in this project unless the team adds one later. The checkpoint documentation should therefore list the public model IDs or the offline copied model folders used in the final run.

### 4.4 Running the Pipeline

Run the notebook in order. The high-level execution flow is:

```text
1. Install dependencies and load configuration.
2. Extract and normalize text from legal PDFs.
3. Parse legal documents into article-level records.
4. Audit parser quality and embedded legal references.
5. Filter valid articles and lock normalized text.
6. Create clause/point-aware retrieval chunks.
7. Store articles and chunks in SQLite.
8. Build or load the legal reference graph.
9. Build or load BM25 and Chroma indexes.
10. Load competition questions.
11. Retrieve, rerank, and select relevant legal articles.
12. Generate Vietnamese legal answers.
13. Validate result format.
14. Export `results.json` and `submission.zip`.
```

For testing, run the single-question and sample-submission sections first. For final output, enable the full submission run:

```python
RUN_FULL_SUBMISSION = True
```

After a successful run, the final files should be available at:

```text
/kaggle/working/results.json
/kaggle/working/submission.zip
```

---

## 5. Submission Format

### 5.1 `results.json` Structure

The final output must be a JSON list. Each item corresponds to one competition question:

```json
[
  {
    "id": 1,
    "question": "",
    "answer": "",
    "relevant_docs": [
      "<document_code>|<document_title>"
    ],
    "relevant_articles": [
      "<document_code>|<document_title>|<article>"
    ]
  }
]
```

Field requirements:

| Field | Requirement |
| --- | --- |
| `id` | Must match the original input question ID |
| `question` | Must keep the original question unchanged |
| `answer` | Must be a Vietnamese legal answer grounded in selected evidence |
| `relevant_docs` | Must contain document references in the required format |
| `relevant_articles` | Must contain article references in the required format |

Document reference format:

```text
<document_code>|<document_title>
```

Article reference format:

```text
<document_code>|<document_title>|<article>
```

Example:

```text
04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4
```

### 5.2 ZIP Submission Structure

The submission file must be a `.zip` file containing `results.json` directly at the root level.

Valid structure:

```text
submission.zip
└── results.json
```

Invalid structure:

```text
submission.zip
└── folder/
    └── results.json
```

The notebook writes the target ZIP file to:

```text
/kaggle/working/submission.zip
```

---

## 6. Private Evaluation Handover Checklist

For private evaluation and reproducibility, prepare the following items:

| Item | Required content | Suggested access method |
| --- | --- | --- |
| Final prediction file | `results.json` and `submission.zip` | Direct upload to competition system |
| Main notebook | `RAGuru(9).ipynb` | Repository, shared archive, or Kaggle notebook link |
| Data package | Legal PDFs, question file, prompt files, optional processed artifacts | Google Drive, OneDrive, Kaggle Dataset, or equivalent |
| Model documentation | Public model IDs, model source links, parameter size, and release-date confirmation | Markdown/PDF |
| Model checkpoints | Public Hugging Face model IDs or offline copied model folders | Hugging Face / shared folder / Kaggle Dataset |
| Dependency list | Package list or `requirements.txt` | Repository root or shared archive |
| Reproduction guide | This handover document and README | Repository root |

All shared links should remain accessible during the organizers' inspection and reproduction period.

---

## 7. Validation Before Submission

Before submitting, verify:

- `results.json` is valid JSON.
- The JSON root is a list.
- Each item contains `id`, `question`, `answer`, `relevant_docs`, and `relevant_articles`.
- Every `id` matches the input question ID.
- Every `question` matches the original input question exactly.
- Every `answer` is written in Vietnamese.
- Every document reference follows `<document_code>|<document_title>`.
- Every article reference follows `<document_code>|<document_title>|<article>`.
- `submission.zip` contains `results.json` directly at the root level.
- No folder wraps `results.json` inside the ZIP file.
- The notebook can be rerun on Kaggle with GPU T4 x2 using the provided data and model access instructions.
- All data and model/checkpoint links are accessible.

The notebook includes runtime validation cells for submission-item structure and final result formatting. These should be executed before uploading the ZIP file.

---

## 8. Known Limitations

- PDF extraction may still be affected by document layout, tables, footers, amendments, or unusual spacing artifacts.
- Legal article parsing may require review for documents with complex embedded references or amendment-style structures.
- Retrieval quality can vary for questions that require multi-domain reasoning or implicit legal context.
- The reference graph is conservative; it is designed to avoid fake article identities, so it may not expand every possible cross-reference.
- Local LLM generation depends on GPU memory and model-loading stability in the Kaggle runtime.
- The system is optimized for competition-style question answering and is not a replacement for legal professionals.

---

## 9. Disclaimer

This project is developed for the R2AI 2026 competition and educational research purposes. The generated answers are for reference only and are not official legal advice. Important legal decisions should be verified against official legal documents and reviewed by qualified legal professionals.
