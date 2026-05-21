# Medical Guidelines Pipeline Comparison & Orchestration Framework

This repository contains a framework for processing medical corpora, embedding chunks into a vector database, and comparing the performance, cost, and output quality of two question-answering pipelines: **Pipeline 1 (LLM-Only)** and **Pipeline 2 (Basic RAG)**. It is designed to evaluate RAG benefits over plain LLM reasoning on complex medical queries.

## 1. Overall Project Motive

The primary goal of this project is to provide a clean, configuration-driven evaluation environment to compare:
- **LLM-Only (Pipeline 1)**: Directly queries a Large Language Model (Mistral 7B) with user questions without context.
- **Basic RAG (Pipeline 2)**: Retrieves semantically relevant context chunks from a local vector store (ChromaDB populated with BGE-small-en-v1.5 embeddings) and augments the LLM prompt with context before inference.

By running both pipelines concurrently and rendering detailed metrics side-by-side (latency, token usage, estimated API costs, cache hits, response content), developers can benchmark the quantitative and qualitative enhancements brought by retrieving medical guidelines context.


## 2. Features

- **Standalone Data Preprocessor**:
  - Token-bound sliding window chunking using `mistral-common` (fallback to `tiktoken` or word-based segmenting).
  - Configurable text cleaning pipeline (strips HTML, filters non-printable characters, normalizes whitespace, removes Gutenberg project headers/footers).
  - Ingests both CSV and JSON formats recursively, mapping metadata such as `question_type`, `evidence`, and source file identifiers.
  - Streaming output writing directly to JSONL to achieve O(1) memory footprint.
- **Vector Indexing Ingestion**:
  - Disables cache during indexing and embeds text chunks locally with the BGE-small-en-v1.5 sentence-transformer model.
  - Upserts documents, IDs, and normalized metadata to ChromaDB.
- **LLM-Only Pipeline (Pipeline 1)**:
  - Fetches responses from Mistral API with exponential backoff retries.
  - Tracks exact completion and prompt tokens via the `mistral-common` tokenizer.
- **Basic RAG Pipeline (Pipeline 2)**:
  - Connects to ChromaDB for similarity search.
  - Integrates Redis-based caching for identical retrieval-query caching.
  - Augments prompts dynamically and requests completion via Mistral API.
- **CLI Orchestrator**:
  - Accept query CLI parameter or run interactively.
  - Batch-testing mode supporting predefined questions.
  - Side-by-side terminal comparison tables showing performance metrics using `rich`.


## 3. Directory Structure

```text
├── config/
│   ├── config.yaml             # Preprocessor configurations
│   ├── llm_only_config.yaml    # Pipeline 1 (LLM-Only) configurations
│   └── basic_rag_config.yaml   # Pipeline 2 (Basic RAG) configurations
├── data/
│   ├── input/                  # Place raw CSV/JSON files here (e.g., medical.json, medical_questions.json)
│   ├── output/                 # Processed preprocessor outputs (e.g., processed_data.jsonl)
│   └── chroma_db/              # Local SQLite-backed Chroma database directory
├── src/
│   ├── preprocessor.py         # Standalone data preprocessing module
│   ├── llm_only/               # Code base for LLM-Only Pipeline
│   │   ├── pipeline.py
│   │   ├── token_counter.py
│   │   └── config_validator.py
│   └── basic_rag/              # Code base for Basic RAG Pipeline
│       ├── pipeline.py
│       ├── vector_store.py
│       ├── embedding_engine.py
│       ├── cache_manager.py
│       └── rag_config_validator.py
├── main.py                     # CLI Pipeline Comparison Orchestrator
├── ingest_to_chroma.py         # DB Ingestion script
├── requirements.txt            # Project dependencies
└── logs/
    ├── preprocessing.log       # Preprocessor run logs
    └── pipeline.log            # Pipeline execution logs
```


## 4. Installation

1. Create and activate a python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## 5. Configuration

### Preprocessor Configuration (`config/config.yaml`)
Configures input paths, required validation fields, chunk size/overlap, text cleaning toggles, and output path/format.

### Pipeline Configurations (`config/llm_only_config.yaml` & `config/basic_rag_config.yaml`)
Configure API details, model names, temperature, retry settings, logging levels, cost estimation rates, and:
- `basic_rag_config.yaml` additionally defines Chroma database collections and local embedding configurations.


## 6. Input Schema & Output Format

### Ingested Preprocessor Inputs
The preprocessor expects datasets in the `data/input/` directory.
- **Corpus Context (e.g. `medical.json`)**: Contains general medical guidelines text.
  ```json
  [ { "corpus_name": "Medical", "context": "..." } ]
  ```
- **Evaluation Questions (e.g. `medical_questions.json`)**:
  ```json
  [
    {
      "id": "Medical-73586ddc",
      "source": "Medical",
      "question": "What is the most common type of skin cancer?",
      "answer": "Basal cell carcinoma is the most common type of skin cancer.",
      "question_type": "Fact Retrieval",
      "evidence": [ "Basal cell carcinoma (BCC) is the most common type of skin cancer." ]
    }
  ]
  ```

### Preprocessor Output Format (`data/output/processed_data.jsonl`)
Each line represents a single chunk:
```json
{
  "chunk_id": "doc_id_chunk_idx",
  "source_doc_id": "original_document_identifier",
  "chunk_text": "cleaned and chunked text string",
  "metadata": {
    "question_type": "Fact Retrieval|Diagnosis|Treatment",
    "evidence": ["array of supporting strings"],
    "source": "original_file_name"
  },
  "token_count": 245,
  "chunk_index": 0,
  "total_chunks_in_doc": 4
}
```

---

## 7. Usage & Running the Framework

Ensure you activate your virtual environment before executing commands:
```bash
source venv/bin/activate
```

### Step 1: Preprocess the Input Data
Run the preprocessor to generate the output chunks. You can restrict the run length using the `--limit` flag:
```bash
python src/preprocessor.py --config config/config.yaml --limit 10
```

### Step 2: Index Chunks into ChromaDB
Ingest the preprocessed JSONL chunks into ChromaDB:
```bash
MISTRAL_API_KEY=dummy python ingest_to_chroma.py --config config/basic_rag_config.yaml --data data/output/processed_data.jsonl --limit 100
```
*(Note: A dummy API key is provided here just to satisfy the pipeline config validation check, as LLM calls are not made during ingestion)*

---

### Step 3: Run the Pipelines

You can run both pipelines either **concurrently (together)** or **individually (separately)**.

#### A. Running Pipelines Together (Concurrent Orchestration)

To run a query against both LLM-Only and Basic RAG pipelines and print the comparison metrics table side-by-side in the terminal:

1. **Dry-Run Mode (Simulation)**:
   Simulates LLM completions without consuming actual network requests or API costs.
   - *Single Query CLI*:
     ```bash
     python main.py --query "What is the most common type of skin cancer?" --dry-run
     ```
   - *Batch-Testing Mode (first 5 questions)*:
     ```bash
     python main.py --limit 5 --dry-run
     ```

2. **Live Mode (API Calls)**:
   Queries the actual Mistral API. Ensure `MISTRAL_API_KEY` is loaded.
   - *Single Query CLI*:
     ```bash
     MISTRAL_API_KEY=your_mistral_api_key_here python main.py --query "What is the most common type of skin cancer?"
     ```
   - *Batch-Testing Mode*:
     ```bash
     MISTRAL_API_KEY=your_mistral_api_key_here python main.py --limit 5
     ```

---

#### B. Running Pipelines Separately

To run only one pipeline, you can use the orchestrator's environment checks. If the API key for a pipeline is missing from the environment, that pipeline is skipped and the available pipeline runs.

1. **Run Pipeline 1 (LLM-Only) Separately**:
   Set `MISTRAL_API_KEY` but provide an invalid or empty configuration database path for Chroma in `config/basic_rag_config.yaml` (which fails Pipeline 2 validation and skips it), or unset Chroma credentials.
   Alternatively, you can load and run Pipeline 1 via python imports directly:
   ```python
   import asyncio
   import yaml
   from src.llm_only import LLMOnlyPipeline

   async def main():
       with open("config/llm_only_config.yaml", "r") as f:
           config = yaml.safe_load(f)
       pipeline = LLMOnlyPipeline(config)
       res = await pipeline.run("What is the most common type of skin cancer?")
       print(res)

   asyncio.run(main())
   ```

2. **Run Pipeline 2 (Basic RAG) Separately**:
   Use python imports to load and run Pipeline 2 directly:
   ```python
   import asyncio
   import yaml
   from src.basic_rag import BasicRAGPipeline

   async def main():
       with open("config/basic_rag_config.yaml", "r") as f:
           config = yaml.safe_load(f)
       pipeline = BasicRAGPipeline(config)
       res = await pipeline.run("What is the most common type of skin cancer?")
       print(res)

   asyncio.run(main())
   ```

## Results
                        BATCH COMPARISON SUMMARY (N=40)                      
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Pipeline 1 (LLM-Only) ┃ Pipeline 2 (Basic RAG) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Total Queries         │                    40 │                     40 │
│ Successful Queries    │                    38 │                     40 │
│ Failed Queries        │                     2 │                      0 │
│ Success Rate          │                95.0%  |                 100.0% |
│ Avg Prompt Tokens     │                  34.5 │                  679.0 │
│ Avg Completion Tokens │                 171.7 │                  128.7 │
│ Avg Total Tokens      │                 206.2 │                  807.7 │
│ Avg Latency (ms)      │                2016.0 │                    0.4 │
│ Total Cost ($)        │             $0.001031 │              $0.004038 │
└───────────────────────┴───────────────────────┴────────────────────────┘