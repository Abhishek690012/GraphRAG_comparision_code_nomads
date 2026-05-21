#!/usr/import/env python3
"""
ChromaDB Ingestion Script for Basic RAG Module

Reads preprocessed chunks from JSONL, batches them, embeds with BGE-M3,
and upserts them into ChromaDB.
"""

import argparse
import json
import logging
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from src.basic_rag.embedding_engine import EmbeddingEngine
from src.basic_rag.vector_store import VectorStoreClient
from src.basic_rag.rag_config_validator import RAGConfigValidator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest chunks into ChromaDB.")
    parser.add_argument("--config", default="config/basic_rag_config.yaml", help="Path to config file.")
    parser.add_argument("--data", default="data/output/processed_data.jsonl", help="Path to input JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to ingest.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    setup_logging()
    args = parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Validate config
    validator = RAGConfigValidator(config)
    validator.validate(dry_run_override=False)

    import time
    
    batch_size = args.batch_size or config["embeddings"].get("batch_size", 16)
    logger.info(f"Initializing embedding engine and vector store (batch_size={batch_size})...")
    logger.info("Downloading/Loading local embedding model. This may take a moment...")

    load_start = time.perf_counter()
    # Disable cache for ingestion
    embedding_engine = EmbeddingEngine(config["embeddings"], cache_manager=None)
    load_time = time.perf_counter() - load_start
    logger.info(f"Model loaded locally in {load_time:.2f} seconds.")
    
    vector_store = VectorStoreClient(config["chroma"])

    logger.info(f"Starting ingestion from {args.data}")
    
    batch_ids = []
    batch_texts = []
    batch_metadatas = []
    
    count = 0
    total_ingested = 0

    with open(args.data, "r") as f:
        for line in f:
            if not line.strip():
                continue
                
            chunk = json.loads(line)
            
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("chunk_text")
            
            if not chunk_id or not text:
                continue
                
            # Build metadata
            metadata = chunk.get("metadata", {})
            metadata["source_doc_id"] = chunk.get("source_doc_id", "")
            metadata["chunk_index"] = chunk.get("chunk_index", 0)
            
            # Chroma metadata values must be str, int, float, or bool.
            # Convert lists/dicts to strings.
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                elif v is not None:
                    clean_metadata[k] = json.dumps(v)
            
            batch_ids.append(chunk_id)
            batch_texts.append(text)
            batch_metadatas.append(clean_metadata)
            
            count += 1
            
            if len(batch_ids) >= batch_size:
                logger.info(f"Encoding batch of {len(batch_texts)} chunks...")
                embeddings = embedding_engine.encode_batch(batch_texts, batch_size=batch_size)
                
                logger.info(f"Upserting batch to ChromaDB...")
                vector_store.upsert(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                
                total_ingested += len(batch_ids)
                logger.info(f"Total ingested: {total_ingested}")
                
                # Clear batches
                batch_ids = []
                batch_texts = []
                batch_metadatas = []
                
            if args.limit and count >= args.limit:
                logger.info(f"Reached limit of {args.limit} chunks.")
                break
                
    # Ingest remaining
    if batch_ids:
        logger.info(f"Encoding final batch of {len(batch_texts)} chunks...")
        embeddings = embedding_engine.encode_batch(batch_texts, batch_size=batch_size)
        vector_store.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        total_ingested += len(batch_ids)
        
    logger.info(f"Ingestion complete. Total chunks ingested: {total_ingested}")
    logger.info(f"ChromaDB collection count: {vector_store.collection_count}")


if __name__ == "__main__":
    main()
