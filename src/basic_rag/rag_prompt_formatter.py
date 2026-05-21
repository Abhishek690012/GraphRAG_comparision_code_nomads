"""
RAG Prompt Formatter for Basic RAG Module

Extends the base PromptFormatter to inject retrieved context into the prompt.
"""

import logging
from typing import Any, Dict, List, Optional

from src.llm_only.prompt_formatter import PromptFormatter
from .vector_store import RetrievalResult

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a helpful medical assistant. Answer the user's question based on the provided context.
If the context does not contain enough information, say so clearly.

### Context:
{context}"""


class RAGPromptFormatter(PromptFormatter):
    """
    Formats queries and retrieved context into the chat message structure required by the API.
    """

    def __init__(self, generation_config: Dict[str, Any]):
        super().__init__(generation_config)
        # Override the default system prompt if it wasn't explicitly set in config,
        # but let config take precedence if provided.
        if "system_prompt" not in generation_config or not generation_config["system_prompt"]:
            self._default_system_prompt = RAG_SYSTEM_PROMPT

    def format_rag_prompt(
        self, 
        query: str, 
        retrieved_chunks: List[RetrievalResult],
        system_prompt: Optional[str] = None, 
        max_context_tokens: int = 4096
    ) -> List[Dict[str, str]]:
        """
        Format a query and context into an OpenAI-compatible messages array.
        
        Args:
            query: The user's input query string.
            retrieved_chunks: List of retrieved chunks.
            system_prompt: Optional override for the system prompt.
            max_context_tokens: Maximum tokens allowed for the context section.
            
        Returns:
            List of message dicts with "role" and "content" keys.
        """
        self._validate_query(query)
        
        effective_system_prompt = (
            system_prompt if system_prompt is not None else self._default_system_prompt
        )
        
        # Assemble context
        context_text = self._assemble_context(retrieved_chunks)
        
        # Inject context into system prompt
        # We assume the system prompt has a {context} placeholder.
        # If not, we append it.
        if "{context}" in effective_system_prompt:
            final_system_prompt = effective_system_prompt.replace("{context}", context_text)
        else:
            final_system_prompt = f"{effective_system_prompt}\n\n### Context:\n{context_text}"
            
        messages = []
        
        if final_system_prompt:
            messages.append({
                "role": "system",
                "content": final_system_prompt.strip(),
            })
            
        messages.append({
            "role": "user",
            "content": query.strip(),
        })
        
        logger.debug(
            f"Formatted RAG prompt: {len(retrieved_chunks)} chunks included."
        )
        return messages

    def _assemble_context(self, chunks: List[RetrievalResult]) -> str:
        """
        Assemble the retrieved chunks into a single context block.
        Includes source attribution for grounding.
        
        TODO: Implement precise token-based truncation here if needed,
        though the orchestrator will catch overall prompt length.
        """
        if not chunks:
            return "No relevant context found."
            
        context_parts = []
        for i, chunk in enumerate(chunks):
            # Format: [Source: doc_id] text...
            part = f"[Source: {chunk.source_doc_id}]\n{chunk.text}"
            context_parts.append(part)
            
        return "\n\n".join(context_parts)
