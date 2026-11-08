"""
BuddyOS Resilient Router

This module provides token-aware model routing with graceful fallback.
Uses LiteLLM for model-agnostic token counting and completion.
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

import litellm
from litellm import token_counter

from core.discovery import discover_available_models, get_model_by_id, get_provider_status
from core.models import ModelInfo, FALLBACK_MODEL_ID, PROVIDER_KEY_ENV, get_small_model_id


# LiteLLM settings shared by every entry point (CLI and Streamlit)
litellm.suppress_debug_info = True
litellm.verbose = False
litellm.drop_params = True  # Strip params a provider doesn't know instead of failing the request

# A stalled provider must not freeze the chat for long before the fallback chain takes over
REQUEST_TIMEOUT_SECONDS = 90

# Below this output budget, hidden reasoning can consume the whole limit and leave an empty reply
SMALL_BUDGET_TOKENS = 1024


# Logging is configured centrally in main.py
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
    content: Optional[str]
    model_used: str
    fallback_occurred: bool = False
    fallback_from: Optional[str] = None
    token_count: int = 0
    tool_calls: Optional[List[Any]] = None


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
        self._provider_status = get_provider_status()
        
        logger.info(f"Router initialized with {len(self.available_models)} available models")
        logger.info(f"Final fallback: {FALLBACK_MODEL_ID}")
    
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
                tokens = int(len((content or "").split()) * 1.3)
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
            provider: Provider name (anthropic, openai, google)
            
        Returns:
            API key string or None if not found
        """
        env_var = PROVIDER_KEY_ENV.get(provider)
        if not env_var:
            return None
        
        return os.getenv(env_var)

    def get_fast_model_for_provider(self, active_model_id: str) -> str:
        """
        Cheapest model of the active model's provider, for background work
        (intent routing, titles, summaries, fact extraction).
        """
        model_info = get_model_by_id(active_model_id)
        small_model_id = get_small_model_id(model_info.provider) if model_info else None
        if small_model_id:
            return small_model_id
        
        # Unknown model: use the free Gemini fallback when its key is available
        if self._provider_status.get("google"):
            return FALLBACK_MODEL_ID
        return active_model_id

    def _get_fallback_chain(self, original_model_id: str) -> List[str]:
        """
        Build fallback chain for given model.
        
        Fallback Priority:
        1. Cheapest (small) model of the same provider
        2. Free Gemini (FALLBACK_MODEL_ID)
        
        Entries that repeat the failing model or an earlier entry, or whose
        provider has no API key configured, are skipped.
        
        Args:
            original_model_id: The model that failed
            
        Returns:
            List of model IDs to try in order
        """
        model_info = get_model_by_id(original_model_id)
        candidates = [
            get_small_model_id(model_info.provider) if model_info else None,
            FALLBACK_MODEL_ID,
        ]
        
        fallback_chain: List[str] = []
        for candidate in candidates:
            if not candidate or candidate == original_model_id or candidate in fallback_chain:
                continue
            candidate_info = get_model_by_id(candidate)
            if candidate_info and self._provider_status.get(candidate_info.provider):
                fallback_chain.append(candidate)
        
        return fallback_chain
    
    def _prepare_call(self, model_info: ModelInfo, kwargs: Dict[str, Any]) -> tuple:
        """Adapt generic call parameters to what this model's API accepts.
        
        Returns:
            (model name to send to LiteLLM, call parameters)
        """
        params = dict(kwargs)
        call_model = model_info.model_id
        
        if not model_info.supports_temperature:
            params.pop("temperature", None)
        
        if model_info.reasoning and params.get("max_tokens") is not None:
            # Hidden thinking tokens count against the output limit; keep room for the visible answer
            params["max_tokens"] = max(params["max_tokens"], SMALL_BUDGET_TOKENS)
        
        if model_info.provider == "openai":
            if model_info.reasoning and params.get("tools"):
                # Chat Completions refuses function tools while reasoning is on; Responses accepts them
                call_model = f"openai/responses/{model_info.model_id}"
            elif "max_tokens" in params:
                # Current OpenAI models reject max_tokens on Chat Completions
                params["max_completion_tokens"] = params.pop("max_tokens")
        
        params.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
        return call_model, params
    
    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Transient provider errors worth retrying on the same model."""
        if isinstance(error, litellm.exceptions.Timeout):
            return False  # A stalled model won't recover within seconds; go to the fallback instead
        if isinstance(error, litellm.exceptions.RateLimitError):
            return "limit: 0" not in str(error)  # A zero quota never recovers by waiting
        return isinstance(error, (
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.APIConnectionError,
        ))
    
    async def _attempt(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        max_retries: int,
        **kwargs
    ) -> CompletionResult:
        """Call one model, retrying transient errors with exponential backoff. Raises on failure."""
        model_info = get_model_by_id(model_id)
        if not model_info:
            raise RouterError(f"Model {model_id} not found in registry")
        
        call_model, params = self._prepare_call(model_info, kwargs)
        api_key = self._get_api_key(model_info.provider)
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempting completion with {model_id} (attempt {attempt + 1}/{max_retries + 1})")
                response = await litellm.acompletion(
                    model=call_model,
                    messages=messages,
                    api_key=api_key,
                    **params
                )
            except Exception as e:
                if attempt < max_retries and self._is_retryable(e):
                    wait_time = 2 ** attempt
                    logger.warning(f"{type(e).__name__} from {model_id}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"{model_id} failed: {type(e).__name__}: {e}")
                raise
            
            message = response.choices[0].message
            content = message.content
            tool_calls = getattr(message, "tool_calls", None)
            
            # Calculate output tokens
            try:
                output_tokens = token_counter(model=model_id, messages=[message.model_dump()])
            except Exception as eval_exc:
                logger.warning(f"Token counting failed for {model_id}: {eval_exc}, using estimate")
                output_tokens = int(len((content or "").split()) * 1.3)
            
            logger.info(f"Completion successful with {model_id}")
            return CompletionResult(
                content=content,
                model_used=model_id,
                token_count=output_tokens,
                tool_calls=tool_calls
            )
        
        raise RouterError(f"No completion attempt was made for {model_id}")
    
    async def get_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_retries: int = 2,
        **kwargs
    ) -> CompletionResult:
        """
        Get LLM completion with automatic fallback.
        
        The selected model gets `max_retries` retries on transient errors; if it still
        fails, the fallback chain (cheapest model of the same provider, then free
        Gemini) is tried with one retry each.
        
        Args:
            model_id: Model identifier
            messages: List of message dicts [{"role": "user", "content": "..."}]
            max_retries: Number of retries on transient errors (default: 2)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            CompletionResult with response and metadata
            
        Raises:
            RouterError: If all models fail including fallback
        """
        original_model_id = model_id
        
        # Validate context window
        try:
            self.validate_payload(model_id, messages, threshold=0.90)
        except ContextWarning as e:
            logger.warning(str(e))
            # Continue anyway, but log warning
        
        # Selected model first, then the fallback chain
        chain = [original_model_id, *self._get_fallback_chain(original_model_id)]
        last_error: Optional[Exception] = None
        
        for index, candidate in enumerate(chain):
            if index == 1:
                logger.warning(f"Primary model {original_model_id} failed, attempting fallback...")
            
            try:
                result = await self._attempt(
                    candidate, messages, max_retries if index == 0 else 1, **kwargs
                )
            except Exception as e:
                last_error = e
                continue
            
            if index > 0:
                result.fallback_occurred = True
                result.fallback_from = original_model_id
                logger.info(f"Fallback successful with {candidate}")
            return result
        
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