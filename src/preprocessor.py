import os
import json
import yaml
import pandas as pd
import tiktoken
import aiofiles
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

class DataPreprocessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.setup_logging()
        try:
            self.tokenizer = tiktoken.get_encoding(self.config['processing']['tokenizer_model'])
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer {self.config['processing']['tokenizer_model']}: {e}. Using fallback word-based token count.")
            self.tokenizer = None

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            filename=self.config['logging']['file'],
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def run(self) -> Dict[str, Any]:
        # Orchestrator
        manifest = {
            'total_input_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'total_chunks': 0,
            'avg_tokens_per_chunk': 0.0,
            'output_files': [],
            'processing_duration': 0.0
        }
        start_time = asyncio.get_event_loop().time()
        try:
            data = self.load_data()
            manifest['total_input_records'] = len(data)
            valid_data = self.validate(data)
            manifest['valid_records'] = len(valid_data)
            manifest['invalid_records'] = manifest['total_input_records'] - manifest['valid_records']
            cleaned_data = [self.clean_text(record) for record in valid_data]
            chunks = []
            for record in cleaned_data:
                record_chunks = self.chunk_text(record)
                for chunk in record_chunks:
                    chunk['metadata'] = self.extract_metadata(record, chunk)
                    chunk['token_count'] = self.count_tokens(chunk['chunk_text'])
                    chunks.append(chunk)
            manifest['total_chunks'] = len(chunks)
            if chunks:
                total_tokens = sum(c['token_count'] for c in chunks)
                manifest['avg_tokens_per_chunk'] = total_tokens / len(chunks)
            if not self.config['output']['dry_run']:
                output_file = await self.save_output(chunks)
                manifest['output_files'].append(output_file)
            manifest['processing_duration'] = asyncio.get_event_loop().time() - start_time
            self.logger.info(f"Processing complete. Manifest: {manifest}")
            return manifest
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        data = []
        context = ""
        files = []
        for path in self.config['input']['paths']:
            files.extend(list(Path(path).rglob('*.json')))
        files.sort(key=lambda f: f.name)
        for file in files:
            try:
                with open(file, 'r') as f:
                    json_data = json.load(f)
                    if 'medical.json' in file.name:
                        if isinstance(json_data, dict):
                            context = json_data.get('context', '')
                        elif isinstance(json_data, list) and json_data:
                            context = json_data[0].get('context', '')
                        self.logger.info(f"Loaded context of length {len(context)}")
                    elif 'medical_questions.json' in file.name:
                        if isinstance(json_data, list):
                            for item in json_data:
                                record = {
                                    'raw_context': context,
                                    'questions': [item.get('question', '')],
                                    'answers': [item.get('answer', '')],
                                    'question_type': item.get('question_type', ''),
                                    'evidence': item.get('evidence', []),
                                    'evidence_relations': item.get('evidence_relations', ''),
                                    'evidence_triples': [],  # Not present
                                    'source_id': item.get('source', ''),
                                    'doc_id': item.get('id', '')
                                }
                                data.append(record)
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
        self.logger.info(f"Loaded {len(data)} records")
        return data

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        required = set(self.config['input']['required_fields'])
        valid = []
        for record in data:
            if required.issubset(set(record.keys())):
                valid.append(record)
            else:
                self.logger.warning(f"Skipping invalid record: missing fields {required - set(record.keys())}")
        return valid

    def clean_text(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # Clean raw_context
        text = record.get('raw_context', '')
        self.logger.info(f"Cleaning text of length {len(text)}")
        if self.config['processing']['text_cleaning']['remove_headers']:
            text = self.remove_gutenberg_headers(text)
        if self.config['processing']['text_cleaning']['remove_html']:
            import re
            text = re.sub(r'<[^>]+>', '', text)
        if self.config['processing']['text_cleaning']['remove_non_printable']:
            text = ''.join(c for c in text if c.isprintable())
        if self.config['processing']['text_cleaning']['normalize_whitespace']:
            text = ' '.join(text.split())
        record['raw_context'] = text
        self.logger.info(f"Cleaned text of length {len(text)}")
        return record

    def remove_gutenberg_headers(self, text: str) -> str:
        # Simple heuristic
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
        self.logger.info(f"Chunking text of length {len(text)} for doc {record['doc_id']}")
        chunk_size = self.config['processing']['chunk_size']
        overlap = self.config['processing']['overlap']
        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunk = {
                'chunk_id': f"{record['doc_id']}_{chunk_index}",
                'source_doc_id': record['doc_id'],
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'total_chunks_in_doc': 0  # Will set later
            }
            chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap
            chunk_index += 1
        # Set total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks_in_doc'] = total
        self.logger.info(f"Created {len(chunks)} chunks for doc {record['doc_id']}")
        return chunks

    def extract_metadata(self, record: Dict[str, Any], chunk: Dict[str, Any]) -> Dict[str, Any]:
        # Map question_type, evidence, etc.
        # For simplicity, copy from record
        metadata = {
            'question_type': record.get('question_type'),
            'evidence': record.get('evidence', []),
            'relations': record.get('evidence_relations', []),
            'triples': record.get('evidence_triples', [])
        }
        return metadata

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate tokens as words / 0.75 (rough estimate for English)
            import re
            words = re.findall(r'\b\w+\b', text)
            return int(len(words) / 0.75)

    async def save_output(self, chunks: List[Dict[str, Any]]) -> str:
        output_path = self.config['output']['path']
        os.makedirs(output_path, exist_ok=True)
        filename = f"processed_data.{self.config['output']['format']}"
        filepath = os.path.join(output_path, filename)
        async with aiofiles.open(filepath, 'w') as f:
            if self.config['output']['format'] == 'jsonl':
                for chunk in chunks:
                    await f.write(json.dumps(chunk) + '\n')
            elif self.config['output']['format'] == 'json':
                await f.write(json.dumps(chunks))
        return filepath