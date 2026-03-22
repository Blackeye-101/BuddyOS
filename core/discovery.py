"""
BuddyOS Model Discovery Module

This module scans the environment for available API keys and returns
a filtered list of supported models based on configured providers.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from core.models import ModelInfo, MODEL_REGISTRY


def get_provider_status() -> Dict[str, bool]:
    """
    Check which API providers are configured.
    
    Returns:
        Dict mapping provider names to their configuration status.
    """
    load_dotenv()
    
    provider_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "xai": "XAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    
    status = {}
    for provider, key_name in provider_keys.items():
        key_value = os.getenv(key_name)
        # Check if key exists and is not the placeholder value
        status[provider] = bool(
            key_value and 
            key_value != f"your_{provider}_key_here" and
            not key_value.startswith("your_")
        )
    
    return status


def discover_available_models(include_paid: bool = True) -> List[ModelInfo]:
    """
    Discover which models are available based on configured API keys.
    
    Args:
        include_paid: If True, include both free and paid models.
                     If False, only include free tier models.
    
    Returns:
        List of ModelInfo objects for models whose provider keys are configured.
    """
    load_dotenv()
    provider_status = get_provider_status()
    
    available_models = []
    
    for provider, models in MODEL_REGISTRY.items():
        if provider_status.get(provider, False):
            for model in models:
                # Filter by tier if include_paid is False
                if include_paid or model.tier == "free":
                    available_models.append(model)
    
    # If no models are available, return Gemini Flash as fallback
    # (assuming it's the free tier option)
    if not available_models:
        # Return free tier options that don't require keys
        fallback = ModelInfo(
            model_id="gemini-3.1-flash",
            provider="google",
            display_name="Gemini 3.1 Flash (Fallback)",
            requires_key="GEMINI_API_KEY",
            description="Default free model - configure API keys for more options",
            tier="free"
        )
        available_models.append(fallback)
    
    return available_models


def get_free_models() -> List[ModelInfo]:
    """
    Get only free tier models that are available.
    
    Returns:
        List of ModelInfo objects for free models only.
    """
    return discover_available_models(include_paid=False)


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """
    Retrieve ModelInfo by model_id.
    
    Args:
        model_id: The unique identifier for the model.
        
    Returns:
        ModelInfo object if found, None otherwise.
    """
    for models in MODEL_REGISTRY.values():
        for model in models:
            if model.model_id == model_id:
                return model
    return None


def get_default_model() -> ModelInfo:
    """
    Get the default model to use.
    
    Priority:
    1. First available configured model
    2. Gemini Flash as fallback
    
    Returns:
        ModelInfo object for the default model.
    """
    available = discover_available_models()
    
    if available:
        return available[0]
    
    # Ultimate fallback
    return ModelInfo(
        model_id="gemini-3.1-flash",
        provider="google",
        display_name="Gemini 3.1 Flash",
        requires_key="GEMINI_API_KEY",
        description="Free tier default model",
        tier="free"
    )


if __name__ == "__main__":
    # Test the discovery module
    print("=== Provider Status ===")
    status = get_provider_status()
    for provider, configured in status.items():
        status_icon = "✅" if configured else "❌"
        print(f"{status_icon} {provider.upper()}: {configured}")
    
    print("\n=== Available Models (All) ===")
    models = discover_available_models(include_paid=True)
    for model in models:
        tier_icon = "💰" if model.tier == "paid" else "🆓"
        print(f"{tier_icon} {model.display_name} ({model.model_id})")
        print(f"   Provider: {model.provider} | Tier: {model.tier}")
        print(f"   Description: {model.description}\n")
    
    print(f"=== Free Models Only ===")
    free_models = get_free_models()
    for model in free_models:
        print(f"🆓 {model.display_name} ({model.model_id})")
        print(f"   Provider: {model.provider}")
        print(f"   Description: {model.description}\n")
    
    print(f"=== Default Model ===")
    default = get_default_model()
    tier_icon = "💰" if default.tier == "paid" else "🆓"
    print(f"{tier_icon} {default.display_name} ({default.model_id})")
    print(f"Tier: {default.tier}")