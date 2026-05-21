import os
import json
import yaml
import pandas as pd
import aiofiles
import asyncio
import logging
import argparse
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

class DataPreprocessor:
    """
    Standalone, deterministic medical data preprocessing module.
    Ingests JSON/CSV inputs, cleans text, chunks by token boundaries,
    extracts metadata, and outputs a single processed_data.jsonl file.
    """
    def __init__(self, config_path: str, limit: Optional[int] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.limit = limit
        self.setup_logging()
        self._init_tokenizer()

    def setup_logging(self):
        log_file = self.config['logging'].get('file', 'logs/preprocessing.log')
        log_level_str = self.config['logging'].get('level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Clear existing root handlers to re-configure cleanly
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.setLevel(log_level)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        
        # Stream (Console) handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(stream_handler)
        
        self.logger = logging.getLogger("preprocessor")
        self.logger.info("Logging configured successfully.")

    def _init_tokenizer(self):
        tokenizer_model = self.config['processing'].get('tokenizer_model', 'mistral-common')
        
        self.tokenizer = None
        self.tokenizer_type = 'fallback'
        
        # Try loading mistral-common first
        if tokenizer_model in ('mistral-common', 'mistral'):
            try:
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
                self.tokenizer = MistralTokenizer.v3().instruct_tokenizer.tokenizer
                self.tokenizer_type = 'mistral'
                self.logger.info("Initialized mistral-common tokenizer (v3).")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load mistral-common tokenizer: {e}. Falling back.")
                
        # Try loading tiktoken
        try:
            import tiktoken
            # Default to cl100k_base if none or mistral failed
            model_name = tokenizer_model if tokenizer_model not in ('mistral-common', 'mistral') else 'cl100k_base'
            self.tokenizer = tiktoken.get_encoding(model_name)
            self.tokenizer_type = 'tiktoken'
            self.logger.info(f"Initialized tiktoken tokenizer with model: {model_name}.")
        except Exception as e:
            self.logger.warning(f"Failed to load tiktoken tokenizer: {e}. Using fallback word-based token count.")
            self.tokenizer = None
            self.tokenizer_type = 'fallback'

    def load_data(self) -> List[Dict[str, Any]]:
        input_paths = self.config['input'].get('paths', ['data/input'])
        formats = self.config['input'].get('formats', ['json', 'csv'])
        
        files = []
        for path_str in input_paths:
            path_obj = Path(path_str)
            if not path_obj.exists():
                self.logger.warning(f"Input path does not exist: {path_str}")
                continue
            if path_obj.is_file():
                files.append(path_obj)
            else:
                for fmt in formats:
                    files.extend(list(path_obj.rglob(f"*.{fmt}")))
                    files.extend(list(path_obj.rglob(f"*.{fmt.upper()}")))
                    
        # Remove duplicates
        files = sorted(list(set(files)), key=lambda f: f.name)
        
        # Pass 1: Load context-only files first
        resolved_contexts = {}
        raw_items = []
        
        for file in files:
            self.logger.info(f"Loading file: {file}")
            try:
                if file.suffix.lower() == '.json':
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        json_data = json.load(f)
                    
                    if isinstance(json_data, dict):
                        if ('context' in json_data or 'raw_context' in json_data or 'raw_text' in json_data) and 'questions' not in json_data and 'question' not in json_data:
                            corpus_name = json_data.get('corpus_name') or file.stem
                            ctx = json_data.get('context') or json_data.get('raw_context') or json_data.get('raw_text')
                            resolved_contexts[corpus_name] = ctx
                            self.logger.info(f"Loaded context '{corpus_name}' of length {len(ctx)}")
                        else:
                            raw_items.append((json_data, file.name))
                    elif isinstance(json_data, list):
                        for item in json_data:
                            if not isinstance(item, dict):
                                continue
                            if ('context' in item or 'raw_context' in item or 'raw_text' in item) and 'questions' not in item and 'question' not in item:
                                corpus_name = item.get('corpus_name') or file.stem
                                ctx = item.get('context') or item.get('raw_context') or item.get('raw_text')
                                resolved_contexts[corpus_name] = ctx
                                self.logger.info(f"Loaded context '{corpus_name}' of length {len(ctx)}")
                            else:
                                raw_items.append((item, file.name))
                
                elif file.suffix.lower() == '.csv':
                    df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
                    for _, row in df.iterrows():
                        row_dict = row.to_dict()
                        raw_items.append((row_dict, file.name))
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                
        # Pass 2: Normalize fields and map questions to context
        data = []
        for item, filename in raw_items:
            fields = self.extract_record_fields(item, filename)
            
            if not fields['raw_context']:
                source = fields['source_id']
                if source in resolved_contexts:
                    fields['raw_context'] = resolved_contexts[source]
                elif len(resolved_contexts) == 1:
                    fields['raw_context'] = list(resolved_contexts.values())[0]
                    
            data.append(fields)
            
        self.logger.info(f"Loaded {len(data)} total records.")
        return data

    def extract_record_fields(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        doc_id = item.get('doc_id') or item.get('id') or item.get('document_ids') or item.get('document_id')
        
        raw_context = (
            item.get('raw_context') or 
            item.get('raw_text') or 
            item.get('context') or 
            item.get('text') or 
            ""
        )
        
        questions = item.get('questions') or item.get('question')
        if isinstance(questions, str):
            questions = [questions]
        elif isinstance(questions, list):
            questions = [str(q) for q in questions if q]
        else:
            questions = []
            
        answers = item.get('answers') or item.get('answer')
        if isinstance(answers, str):
            answers = [answers]
        elif isinstance(answers, list):
            answers = [str(a) for a in answers if a]
        else:
            answers = []
            
        question_type = item.get('question_type') or ""
        
        evidence = item.get('evidence') or item.get('evidence_arrays') or item.get('evidence_relations')
        if isinstance(evidence, str):
            evidence = [evidence]
        elif isinstance(evidence, list):
            evidence = [str(e) for e in evidence if e]
        else:
            evidence = []
            
        source_id = item.get('source_id') or item.get('source') or item.get('corpus_name') or item.get('source_identifiers') or filename
        
        if doc_id is None:
            import hashlib
            content_str = "".join(questions) + "".join(answers) + raw_context[:100]
            doc_id = f"doc_{hashlib.md5(content_str.encode('utf-8')).hexdigest()[:8]}"
            
        return {
            'doc_id': str(doc_id),
            'raw_context': str(raw_context),
            'questions': questions,
            'answers': answers,
            'question_type': str(question_type),
            'evidence': evidence,
            'source_id': str(source_id)
        }

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for record in data:
            if not record.get('doc_id'):
                self.logger.warning("Skipping record: missing doc_id")
                continue
            if not record.get('raw_context') or not record['raw_context'].strip():
                self.logger.warning(f"Skipping record {record.get('doc_id')}: empty raw_context")
                continue
            valid.append(record)
        return valid

    def clean_text(self, record: Dict[str, Any]) -> Dict[str, Any]:
        text = record.get('raw_context', '')
        
        if self.config['processing']['text_cleaning'].get('remove_html', False):
            text = re.sub(r'<[^>]+>', '', text)
            
        if self.config['processing']['text_cleaning'].get('remove_non_printable', False):
            text = ''.join(c for c in text if c.isprintable() or c in ('\n', '\r', '\t'))
            
        if self.config['processing']['text_cleaning'].get('remove_headers', False):
            text = self.remove_gutenberg_headers(text)
            
        if self.config['processing']['text_cleaning'].get('normalize_whitespace', False):
            text = ' '.join(text.split())
            
        record['raw_context'] = text
        return record

    def remove_gutenberg_headers(self, text: str) -> str:
        lines = text.split('\n')
        start = 0
        end = len(lines)
        for i, line in enumerate(lines):
            if '*** START OF' in line.upper():
                start = i + 1
                break
        for i in range(len(lines) - 1, -1, -1):
            if '*** END OF' in lines[i].upper():
                end = i
                break
        return '\n'.join(lines[start:end])

    def chunk_text(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = record['raw_context']
        chunk_size = self.config['processing'].get('chunk_size', 512)
        overlap = self.config['processing'].get('overlap', 50)
        
        token_ids = None
        if self.tokenizer_type == 'mistral':
            token_ids = self.tokenizer.encode(text, bos=False, eos=False)
        elif self.tokenizer_type == 'tiktoken':
            token_ids = self.tokenizer.encode(text)
            
        chunks = []
        if token_ids is not None:
            start = 0
            chunk_index = 0
            while start < len(token_ids):
                end = min(start + chunk_size, len(token_ids))
                chunk_tokens = token_ids[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunk = {
                    'chunk_id': f"{record['doc_id']}_{chunk_index}",
                    'source_doc_id': record['doc_id'],
                    'chunk_text': chunk_text,
                    'chunk_index': chunk_index,
                    'total_chunks_in_doc': 0,
                    'token_count': len(chunk_tokens)
                }
                chunks.append(chunk)
                
                if end >= len(token_ids):
                    break
                start = max(end - overlap, start + 1)
                chunk_index += 1
        else:
            words = text.split()
            start = 0
            chunk_index = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                
                chunk = {
                    'chunk_id': f"{record['doc_id']}_{chunk_index}",
                    'source_doc_id': record['doc_id'],
                    'chunk_text': chunk_text,
                    'chunk_index': chunk_index,
                    'total_chunks_in_doc': 0,
                    'token_count': len(chunk_words)
                }
                chunks.append(chunk)
                if end >= len(words):
                    break
                start = max(end - overlap, start + 1)
                chunk_index += 1
                
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks_in_doc'] = total
            
        return chunks

    def extract_metadata(self, record: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'question_type': record.get('question_type', ''),
            'evidence': record.get('evidence', []),
            'source': record.get('source_id', '')
        }

    async def run(self) -> Dict[str, Any]:
        manifest = {
            'total_input_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'total_chunks': 0,
            'avg_tokens_per_chunk': 0.0,
            'output_files': [],
            'processing_duration': 0.0
        }
        start_time = time.perf_counter()
        
        try:
            self.logger.info("Starting load_data phase...")
            data = self.load_data()
            manifest['total_input_records'] = len(data)
            
            self.logger.info("Starting validation phase...")
            valid_data = self.validate(data)
            manifest['valid_records'] = len(valid_data)
            manifest['invalid_records'] = manifest['total_input_records'] - manifest['valid_records']
            
            if self.limit is not None and self.limit > 0:
                self.logger.info(f"Limiting execution to the first {self.limit} records.")
                valid_data = valid_data[:self.limit]
                
            output_dir = self.config['output'].get('path', 'data/output')
            os.makedirs(output_dir, exist_ok=True)
            output_format = self.config['output'].get('format', 'jsonl')
            filename = f"processed_data.{output_format}"
            filepath = os.path.join(output_dir, filename)
            
            total_chunks = 0
            total_tokens = 0
            dry_run = self.config['output'].get('dry_run', False)
            
            self.logger.info(f"Starting chunking and writing phase. Output file: {filepath} (dry_run: {dry_run})")
            
            f_out = None
            if not dry_run:
                f_out = await aiofiles.open(filepath, 'w', encoding='utf-8')
                
            try:
                all_chunks = [] if (dry_run or output_format == 'json') else None
                
                for idx, record in enumerate(valid_data):
                    if idx > 0 and idx % 100 == 0:
                        self.logger.info(f"Processed {idx} records, generated {total_chunks} chunks.")
                        
                    cleaned_record = self.clean_text(record)
                    record_chunks = self.chunk_text(cleaned_record)
                    
                    for chunk in record_chunks:
                        chunk['metadata'] = self.extract_metadata(cleaned_record, chunk)
                        total_chunks += 1
                        total_tokens += chunk['token_count']
                        
                        if not dry_run:
                            if output_format == 'jsonl':
                                await f_out.write(json.dumps(chunk) + '\n')
                            elif output_format == 'json':
                                all_chunks.append(chunk)
                        elif dry_run:
                            all_chunks.append(chunk)
                            
                if not dry_run and output_format == 'json':
                    await f_out.write(json.dumps(all_chunks))
                    
            finally:
                if f_out:
                    await f_out.close()
                    
            manifest['total_chunks'] = total_chunks
            if total_chunks > 0:
                manifest['avg_tokens_per_chunk'] = total_tokens / total_chunks
                
            if not dry_run:
                manifest['output_files'].append(filepath)
                
            manifest['processing_duration'] = time.perf_counter() - start_time
            self.logger.info(f"Processing complete. Manifest: {manifest}")
            return manifest
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Medical Data Preprocessor")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process.")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(config_path=args.config, limit=args.limit)
    asyncio.run(preprocessor.run())