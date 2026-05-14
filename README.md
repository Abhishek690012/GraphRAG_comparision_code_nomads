# Data Preprocessing Module

This module provides a standalone, modular data preprocessing pipeline for CSV/JSON datasets, designed for downstream vector indexing and graph ingestion.

## Features

- Ingests raw CSV and JSON files
- Validates schema and handles missing/invalid data
- Cleans text (removes headers, HTML, non-printable chars, normalizes whitespace)
- Chunks text into configurable sizes with overlap
- Extracts and structures metadata (evidence relations/triples)
- Computes token counts using configurable tokenizer
- Outputs standardized JSONL/JSON format
- Async-compatible I/O
- Configuration-driven via YAML
- Comprehensive logging and manifest generation

## Directory Structure

- `src/`: Source code
- `config/`: Configuration files
- `data/input/`: Place your input CSV/JSON files here
- `data/output/`: Processed output will be saved here
- `logs/`: Log files

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config/config.yaml` to set parameters:

- Input paths and required fields
- Processing options (chunk size, overlap, tokenizer)
- Output format and dry-run mode
- Logging level

## Usage

1. Place your data files in `data/input/`.
2. Run the preprocessor:
   ```bash
   python main.py
   ```
3. Check `data/output/` for processed files and `logs/` for logs.

## Input Schema

Files must contain the following fields:
- `raw_context`: The main text to process
- `questions`: Associated questions
- `answers`: Answers
- `question_type`: Type of question
- `evidence`: Evidence arrays
- `evidence_relations`: Relations
- `evidence_triples`: Triples
- `source_id`: Source identifier
- `doc_id`: Document ID

## Output Format

Each output record contains:
- `chunk_id`: Unique chunk identifier
- `source_doc_id`: Original document ID
- `chunk_text`: Processed text chunk
- `metadata`: Extracted metadata
- `token_count`: Token count for the chunk
- `chunk_index`: Index within document
- `total_chunks_in_doc`: Total chunks in document

## Manifest

After processing, a manifest is printed with statistics:
- Total input records
- Valid/invalid counts
- Total chunks
- Average tokens per chunk
- Output files
- Processing duration