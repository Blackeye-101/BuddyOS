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


@dataclass(frozen=True)
class ContextPreset:
    """UI/CLI-selectable context size: what each request carries and when old turns get summarized."""
    label: str
    window_tokens: int  # Rolling summarization starts at 75% of this
    history_budget: int  # Max tokens of history sent per request
    recent_messages: int  # Newest messages always sent, even over budget


# Window >> history budget, otherwise the 75% summarization trigger would fire every turn
CONTEXT_PRESETS: Dict[str, ContextPreset] = {
    "small": ContextPreset("Small", window_tokens=8_000, history_budget=2_000, recent_messages=3),
    "large": ContextPreset("Large", window_tokens=32_000, history_budget=6_000, recent_messages=8),
}
DEFAULT_CONTEXT_SIZE = "large"


def resolve_context_window(model: Optional[ModelInfo], context_size: str) -> int:
    """Preset window, capped by what the model actually supports."""
    window = CONTEXT_PRESETS[context_size].window_tokens
    return min(window, model.context_window) if model else window


DEFAULT_MODEL_ID = "gemini/gemini-3.8-flash"  # UI/CLI default; balanced, free tier
FALLBACK_MODEL_ID = "gemini/gemini-3.5-flash-lite"  # Router last resort and Gemini background tasks; cheapest, free tier

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
            max_tokens=128_000
        ),
        ModelInfo(
            model_id="claude-sonnet-5-5",
            provider="anthropic",
            display_name="Claude Sonnet 5.5 (Balanced)",
            requires_key="ANTHROPIC_API_KEY",
            description="Best mix of speed, intelligence and cost",
            tier="paid",
            context_window=1_000_000,
            max_tokens=128_000
        ),
        ModelInfo(
            model_id="claude-opus-5-5",
            provider="anthropic",
            display_name="Claude Opus 5.5 (Large)",
            requires_key="ANTHROPIC_API_KEY",
            description="Highest capability for long-running agentic and knowledge work; most expensive",
            tier="paid",
            context_window=1_000_000,
            max_tokens=128_000
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
            context_window=1_050_000,
            max_tokens=128_000
        ),
        ModelInfo(
            model_id="gpt-6.1-sol",
            provider="openai",
            display_name="GPT-6.1 Sol (Balanced)",
            requires_key="OPENAI_API_KEY",
            description="Near-flagship performance at a much lower cost",
            tier="paid",
            context_window=1_050_000,
            max_tokens=128_000
        ),
        ModelInfo(
            model_id="gpt-6-astra",
            provider="openai",
            display_name="GPT-6 Astra (Large)",
            requires_key="OPENAI_API_KEY",
            description="Flagship for the most demanding reasoning and coding; most expensive",
            tier="paid",
            context_window=1_050_000,
            max_tokens=128_000
        ),
    ],
    "xai": [
        ModelInfo(
            model_id="grok-4-20",
            provider="xai",
            display_name="Grok 4-20",
            requires_key="XAI_API_KEY",
            description="xAI's Grok model",
            tier="paid",
            context_window=128000,  # 128K tokens (estimated)
            max_tokens=8192  # 8K max output (estimated)
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
            max_tokens=65_536
        ),
        ModelInfo(
            model_id="gemini/gemini-3.8-flash",
            provider="google",
            display_name="Gemini 3.8 Flash (Balanced)",
            requires_key="GEMINI_API_KEY",
            description="Most intelligent Flash; best cost/performance (free tier available)",
            tier="free",
            context_window=1_048_576,
            max_tokens=65_536
        ),
        ModelInfo(
            model_id="gemini/gemini-4-argon",
            provider="google",
            display_name="Gemini 4 Argon (Large)",
            requires_key="GEMINI_API_KEY",
            description="Flagship reasoning model; not on the free tier, most expensive",
            tier="paid",
            context_window=1_048_576,
            max_tokens=1_048_576
        ),
    ],
    "groq": [
        ModelInfo(
            model_id="groq/openai/gpt-oss-120b",
            provider="groq",
            display_name="GPT-OSS-120B",
            requires_key="GROQ_API_KEY",
            description="Open-source model on Groq infrastructure",
            tier="paid",
            context_window=131072,  # 131K tokens (estimated)
            max_tokens=8192  # 8K max output (estimated)
        ),
        ModelInfo(
            model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            provider="groq",
            display_name="Llama 4 Scout 17B",
            requires_key="GROQ_API_KEY",
            description="Meta's Llama 4 Scout on Groq",
            tier="paid",
            context_window=131072,  # 131K tokens (estimated)
            max_tokens=8192  # 8K max output (estimated)
        ),
    ],
    "openrouter": [
        ModelInfo(
            model_id="openrouter/auto",
            provider="openrouter",
            display_name="OpenRouter Auto",
            requires_key="OPENROUTER_API_KEY",
            description="Automatic model selection",
            tier="free",
            context_window=128000,  # 128K tokens (varies by routed model)
            max_tokens=16000  # 16K max output (varies by routed model)
        ),
        # DeepSeek Free Models
        ModelInfo(
            model_id="openrouter/deepseek/deepseek-r1:free",
            provider="openrouter",
            display_name="DeepSeek R1 (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="671B reasoning model, performance on par with OpenAI o1",
            tier="free",
            context_window=128000,  # 128K tokens
            max_tokens=32768  # 32K max output
        ),
        ModelInfo(
            model_id="openrouter/deepseek/deepseek-r1-distill-qwen-32b:free",
            provider="openrouter",
            display_name="DeepSeek R1 Distill Qwen 32B (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="32B distilled reasoning model, outperforms o1-mini",
            tier="free",
            context_window=128000,  # 128K tokens
            max_tokens=16000  # 16K max output
        ),
        ModelInfo(
            model_id="openrouter/deepseek/deepseek-r1-distill-qwen-14b:free",
            provider="openrouter",
            display_name="DeepSeek R1 Distill Qwen 14B (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="14B compact reasoning model",
            tier="free",
            context_window=128000,  # 128K tokens
            max_tokens=16000  # 16K max output
        ),
        # Qwen Free Models
        ModelInfo(
            model_id="openrouter/qwen/qwen3-coder:free",
            provider="openrouter",
            display_name="Qwen3 Coder 480B (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="480B MoE coding model, 35B active parameters",
            tier="free",
            context_window=262144,  # 262K tokens
            max_tokens=16000  # 16K max output
        ),
        ModelInfo(
            model_id="openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
            provider="openrouter",
            display_name="Qwen3 Next 80B (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="80B fast instruction model for RAG and agents",
            tier="free",
            context_window=262144,  # 262K tokens
            max_tokens=16000  # 16K max output
        ),
    ],
}