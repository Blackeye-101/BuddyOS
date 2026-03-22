"""
BuddyOS Resilient Router

This module provides token-aware model routing with graceful fallback.
Uses LiteLLM for model-agnostic token counting and completion.
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

import litellm
from litellm import token_counter, completion

from core.discovery import discover_available_models, get_model_by_id, get_free_models
from core.models import ModelInfo


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextWarning(Exception):
    """Raised when payload exceeds 90% of model's context window."""
    pass


class RouterError(Exception):
    """Raised when all models fail including fallback chain."""
    pass


@dataclass
class CompletionResult:
    """Result of a completion request."""
    content: str
    model_used: str
    fallback_occurred: bool = False
    fallback_from: Optional[str] = None
    token_count: int = 0


class BuddyRouter:
    """
    Resilient router for multi-provider model access via LiteLLM.
    
    Features:
    - Token validation before sending requests
    - Retry logic for transient failures
    - Fallback chain for permanent failures
    - Model-agnostic token counting
    """
    
    def __init__(self):
        """Initialize router with available models from discovery."""
        load_dotenv()
        
        self.available_models = discover_available_models(include_paid=True)
        self.free_models = get_free_models()
        
        # Ultimate fallback
        self.ultimate_fallback = "gemini-3.1-flash"
        
        logger.info(f"Router initialized with {len(self.available_models)} available models")
        logger.info(f"Ultimate fallback: {self.ultimate_fallback}")
    
    def validate_payload(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        threshold: float = 0.90
    ) -> int:
        """
        Validate that payload doesn't exceed context window.
        
        Args:
            model_id: Model identifier
            messages: List of message dicts
            threshold: Warning threshold (default 0.90 = 90%)
            
        Returns:
            Total token count
            
        Raises:
            ContextWarning: If token count exceeds threshold
        """
        model_info = get_model_by_id(model_id)
        if not model_info:
            logger.warning(f"Model {model_id} not found in registry, skipping validation")
            return 0
        
        # Calculate total tokens across all messages
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            try:
                # Use LiteLLM's token counter (model-agnostic)
                tokens = token_counter(model=model_id, text=content)
                total_tokens += tokens
            except Exception as e:
                logger.warning(f"Token counting failed for {model_id}: {e}, using estimate")
                # Fallback: rough estimate (1 token ≈ 0.75 words)
                tokens = int(len(content.split()) * 1.3)
                total_tokens += tokens
        
        # Check against context window
        context_limit = model_info.context_window
        if total_tokens >= (context_limit * threshold):
            raise ContextWarning(
                f"Token count ({total_tokens}) exceeds {int(threshold*100)}% "
                f"of context window ({context_limit}) for {model_id}"
            )
        
        logger.debug(f"Token validation passed: {total_tokens}/{context_limit} tokens")
        return total_tokens
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """
        Retrieve API key from environment for given provider.
        
        Args:
            provider: Provider name (anthropic, openai, google, etc.)
            
        Returns:
            API key string or None if not found
        """
        key_mapping = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
            "google": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        
        env_var = key_mapping.get(provider)
        if not env_var:
            return None
        
        return os.getenv(env_var)
    
    def _find_free_model_same_provider(self, original_model_id: str) -> Optional[str]:
        """
        Find a free model from the same provider.
        
        Args:
            original_model_id: The model that failed
            
        Returns:
            Free model ID from same provider, or None
        """
        original_model = get_model_by_id(original_model_id)
        if not original_model:
            return None
        
        original_provider = original_model.provider
        
        # Find free models from same provider
        for model in self.free_models:
            if model.provider == original_provider:
                logger.info(f"Found free model from same provider: {model.model_id}")
                return model.model_id
        
        return None
    
    def _get_fallback_chain(self, original_model_id: str) -> List[str]:
        """
        Build fallback chain for given model.
        
        Fallback Priority:
        1. Same provider, free tier model (if available)
        2. OpenRouter free models (DeepSeek R1, Qwen)
        3. Gemini 3.1 Flash (ultimate fallback)
        
        Args:
            original_model_id: The model that failed
            
        Returns:
            List of model IDs to try in order
        """
        fallback_chain = []
        
        # 1. Try free model from same provider
        same_provider_free = self._find_free_model_same_provider(original_model_id)
        if same_provider_free and same_provider_free != original_model_id:
            fallback_chain.append(same_provider_free)
        
        # 2. Try OpenRouter free models
        openrouter_models = [
            "deepseek/deepseek-r1:free",  # Powerful reasoning model
            "qwen/qwen3-coder:free",       # Good for code/technical tasks
        ]
        
        for model_id in openrouter_models:
            model_info = get_model_by_id(model_id)
            if model_info and model_id not in fallback_chain:
                # Check if OpenRouter key is available
                if self._get_api_key("openrouter"):
                    fallback_chain.append(model_id)
        
        # 3. Ultimate fallback: Gemini 3.1 Flash
        if self.ultimate_fallback not in fallback_chain:
            fallback_chain.append(self.ultimate_fallback)
        
        return fallback_chain
    
    async def get_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        **kwargs
    ) -> CompletionResult:
        """
        Get LLM completion with automatic fallback.
        
        Args:
            model_id: Model identifier
            messages: List of message dicts [{"role": "user", "content": "..."}]
            max_retries: Number of retries on rate limit (default: 2)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            CompletionResult with response and metadata
            
        Raises:
            RouterError: If all models fail including fallback
        """
        original_model_id = model_id
        
        # Validate context window
        try:
            token_count = self.validate_payload(model_id, messages, threshold=0.90)
        except ContextWarning as e:
            logger.warning(str(e))
            # Continue anyway, but log warning
            token_count = 0
        
        # Try primary model with retries
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempting completion with {model_id} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Get API key for provider
                model_info = get_model_by_id(model_id)
                if not model_info:
                    raise RouterError(f"Model {model_id} not found in registry")
                
                api_key = self._get_api_key(model_info.provider)
                
                # Call LiteLLM
                response = completion(
                    model=model_id,
                    messages=messages,
                    api_key=api_key,
                    **kwargs
                )
                
                # Extract content
                content = response.choices[0].message.content
                
                # Calculate output tokens
                try:
                    output_tokens = token_counter(model=model_id, text=content)
                except Exception:
                    output_tokens = int(len(content.split()) * 1.3)
                
                logger.info(f"Completion successful with {model_id}")
                
                return CompletionResult(
                    content=content,
                    model_used=model_id,
                    fallback_occurred=False,
                    token_count=output_tokens
                )
                
            except litellm.exceptions.RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit hit for {model_id}, attempt {attempt + 1}")
                if attempt < max_retries:
                    # Exponential backoff
                    import asyncio
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached for {model_id}")
                    break
                    
            except (
                litellm.exceptions.AuthenticationError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.APIError
            ) as e:
                last_error = e
                logger.error(f"Permanent error with {model_id}: {type(e).__name__}: {e}")
                break  # Don't retry on permanent errors
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error with {model_id}: {e}")
                break
        
        # Primary model failed, try fallback chain
        logger.warning(f"Primary model {model_id} failed, attempting fallback...")
        fallback_chain = self._get_fallback_chain(original_model_id)
        
        for fallback_model in fallback_chain:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                
                model_info = get_model_by_id(fallback_model)
                if not model_info:
                    logger.warning(f"Fallback model {fallback_model} not in registry")
                    continue
                
                api_key = self._get_api_key(model_info.provider)
                
                response = completion(
                    model=fallback_model,
                    messages=messages,
                    api_key=api_key,
                    **kwargs
                )
                
                content = response.choices[0].message.content
                
                try:
                    output_tokens = token_counter(model=fallback_model, text=content)
                except Exception:
                    output_tokens = int(len(content.split()) * 1.3)
                
                logger.info(f"Fallback successful with {fallback_model}")
                
                return CompletionResult(
                    content=content,
                    model_used=fallback_model,
                    fallback_occurred=True,
                    fallback_from=original_model_id,
                    token_count=output_tokens
                )
                
            except Exception as e:
                logger.error(f"Fallback model {fallback_model} failed: {e}")
                continue
        
        # All models failed
        raise RouterError(
            f"All models failed including fallback chain. "
            f"Original model: {original_model_id}, "
            f"Last error: {last_error}"
        )
    
    def list_available_models(self) -> List[ModelInfo]:
        """
        Return all available models from discovery.
        
        Returns:
            List of ModelInfo objects
        """
        return self.available_models
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get ModelInfo for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelInfo object or None if not found
        """
        return get_model_by_id(model_id)


# ============================================
# Convenience Functions
# ============================================

def create_router() -> BuddyRouter:
    """
    Create and initialize a BuddyRouter instance.
    
    Returns:
        Initialized BuddyRouter
    """
    return BuddyRouter()