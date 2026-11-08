"""
BuddyOS Model Registry

Central registry of all supported models across providers.
This file contains the source of truth for available models.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a supported model."""
    model_id: str
    provider: str
    display_name: str
    requires_key: str
    description: Optional[str] = None
    tier: str = "free"  # "free" or "paid" - indicates if model requires paid subscription
    context_window: int = 128000  # Total input context window size in tokens
    max_tokens: int = 4096  # Maximum output tokens per response
    size: str = "balanced"  # "small" | "balanced" | "large": cost/performance class within its provider
    reasoning: bool = False  # Spends hidden tokens thinking, so it needs output-token headroom
    supports_temperature: bool = True  # Claude 5.5 and GPT-6 reject custom temperature values


@dataclass(frozen=True)
class ContextPreset:
    """UI/CLI-selectable context size: how much history each request carries and when old turns get summarized."""
    label: str
    window_tokens: int  # Rolling summarization starts at 75% of this
    history_budget: int  # Max tokens of history sent per request; the latest exchange is always sent
    recent_messages: int  # Newest messages considered per request; even, so history starts on a user turn


# Window >> history budget, otherwise the 75% summarization trigger would fire every turn
CONTEXT_PRESETS: Dict[str, ContextPreset] = {
    "small": ContextPreset("Small", window_tokens=8_000, history_budget=2_000, recent_messages=4),
    "large": ContextPreset("Large", window_tokens=32_000, history_budget=6_000, recent_messages=8),
}
DEFAULT_CONTEXT_SIZE = "small"


def get_context_preset(context_size: str) -> ContextPreset:
    """Preset for a UI/CLI selection, falling back to the default for unknown values."""
    return CONTEXT_PRESETS.get(context_size) or CONTEXT_PRESETS[DEFAULT_CONTEXT_SIZE]


def resolve_context_window(model: Optional[ModelInfo], context_size: str) -> int:
    """Preset window, capped by what the model actually supports."""
    window = get_context_preset(context_size).window_tokens
    return min(window, model.context_window) if model else window


# Environment variable holding each provider's API key
PROVIDER_KEY_ENV: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}

DEFAULT_MODEL_ID = "gemini/gemini-3.5-flash-lite"  # Gemini 3.8 Flash took 45-120s per reply on a free-tier key (2026-11-08)
FALLBACK_MODEL_ID = "gemini/gemini-3.5-flash-lite"  # Router's last resort; free tier

# Model Registry - small / balanced / large per provider, verified against provider docs on 2026-10-30
MODEL_REGISTRY: Dict[str, List[ModelInfo]] = {
    "anthropic": [
        ModelInfo(
            model_id="claude-haiku-5-5",
            provider="anthropic",
            display_name="Claude Haiku 5.5 (Small)",
            requires_key="ANTHROPIC_API_KEY",
            description="Cheapest Claude; high-volume, latency-sensitive tasks",
            tier="paid",
            context_window=1_000_000,
            max_tokens=128_000,
            size="small",
            supports_temperature=False
        ),
        ModelInfo(
            model_id="claude-sonnet-5-5",
            provider="anthropic",
            display_name="Claude Sonnet 5.5 (Balanced)",
            requires_key="ANTHROPIC_API_KEY",
            description="Best mix of speed, intelligence and cost",
            tier="paid",
            context_window=1_000_000,
            max_tokens=128_000,
            size="balanced",
            supports_temperature=False
        ),
        ModelInfo(
            model_id="claude-opus-5-5",
            provider="anthropic",
            display_name="Claude Opus 5.5 (Large)",
            requires_key="ANTHROPIC_API_KEY",
            description="Highest capability for long-running agentic and knowledge work; most expensive",
            tier="paid",
            context_window=1_000_000,
            max_tokens=128_000,
            size="large",
            supports_temperature=False
        ),
    ],
    "openai": [
        ModelInfo(
            model_id="gpt-6-luna",
            provider="openai",
            display_name="GPT-6 Luna (Small)",
            requires_key="OPENAI_API_KEY",
            description="Most efficient GPT-6; focused, high-volume tasks",
            tier="paid",
            context_window=922_000,
            max_tokens=128_000,
            size="small",
            reasoning=True,
            supports_temperature=False
        ),
        ModelInfo(
            model_id="gpt-6.1-sol",
            provider="openai",
            display_name="GPT-6.1 Sol (Balanced)",
            requires_key="OPENAI_API_KEY",
            description="Near-flagship performance at a much lower cost",
            tier="paid",
            context_window=922_000,
            max_tokens=128_000,
            size="balanced",
            reasoning=True,
            supports_temperature=False
        ),
        ModelInfo(
            model_id="gpt-6-astra",
            provider="openai",
            display_name="GPT-6 Astra (Large)",
            requires_key="OPENAI_API_KEY",
            description="Flagship for the most demanding reasoning and coding; most expensive",
            tier="paid",
            context_window=922_000,
            max_tokens=128_000,
            size="large",
            reasoning=True,
            supports_temperature=False
        ),
    ],
    "google": [
        ModelInfo(
            model_id="gemini/gemini-3.5-flash-lite",
            provider="google",
            display_name="Gemini 3.5 Flash-Lite (Small)",
            requires_key="GEMINI_API_KEY",
            description="Cheapest Gemini; high-volume agentic tasks (free tier available)",
            tier="free",
            context_window=1_048_576,
            max_tokens=65_536,
            size="small"
        ),
        ModelInfo(
            model_id="gemini/gemini-3.8-flash",
            provider="google",
            display_name="Gemini 3.8 Flash (Balanced)",
            requires_key="GEMINI_API_KEY",
            description="Most intelligent Flash; best cost/performance (free tier available)",
            tier="free",
            context_window=1_048_576,
            max_tokens=65_536,
            size="balanced",
            reasoning=True
        ),
        ModelInfo(
            model_id="gemini/gemini-4-argon",
            provider="google",
            display_name="Gemini 4 Argon (Large)",
            requires_key="GEMINI_API_KEY",
            description="Flagship reasoning model; paid plan required, most expensive",
            tier="paid",
            context_window=1_048_576,
            max_tokens=1_048_576,
            size="large",
            reasoning=True
        ),
    ],
}


def get_small_model_id(provider: str) -> Optional[str]:
    """Cheapest model of a provider; used for background tasks and as the first fallback."""
    return next((m.model_id for m in MODEL_REGISTRY.get(provider, []) if m.size == "small"), None)