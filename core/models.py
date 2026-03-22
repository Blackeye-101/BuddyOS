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


# Model Registry - March 2026 supported models
MODEL_REGISTRY: Dict[str, List[ModelInfo]] = {
    "anthropic": [
        ModelInfo(
            model_id="claude-opus-4-6-20250514",
            provider="anthropic",
            display_name="Claude 4.6 Opus",
            requires_key="ANTHROPIC_API_KEY",
            description="Most capable model for complex tasks",
            tier="paid",
            context_window=1000000,  # 1M tokens
            max_tokens=128000  # 128K max output
        ),
        ModelInfo(
            model_id="claude-sonnet-4-6-20250514",
            provider="anthropic",
            display_name="Claude 4.6 Sonnet",
            requires_key="ANTHROPIC_API_KEY",
            description="Balanced performance and speed",
            tier="paid",
            context_window=1000000,  # 1M tokens
            max_tokens=64000  # 64K max output
        ),
        ModelInfo(
            model_id="claude-sonnet-4-5-20250514",
            provider="anthropic",
            display_name="Claude Sonnet 4.5",
            requires_key="ANTHROPIC_API_KEY",
            description="Previous generation Sonnet model",
            tier="paid",
            context_window=200000,  # 200K tokens (1M in beta)
            max_tokens=64000  # 64K max output
        ),
        ModelInfo(
            model_id="claude-3-7-sonnet-20250219",
            provider="anthropic",
            display_name="Claude 3.7 Sonnet",
            requires_key="ANTHROPIC_API_KEY",
            description="Claude 3.x series Sonnet",
            tier="paid",
            context_window=200000,  # 200K tokens
            max_tokens=16000  # 16K max output
        ),
    ],
    "openai": [
        ModelInfo(
            model_id="gpt-5",
            provider="openai",
            display_name="GPT-5",
            requires_key="OPENAI_API_KEY",
            description="OpenAI's flagship model",
            tier="paid",
            context_window=1050000,  # 1.05M tokens (GPT-5.4)
            max_tokens=16000  # 16K max output
        ),
        ModelInfo(
            model_id="gpt-4o",
            provider="openai",
            display_name="GPT-4o",
            requires_key="OPENAI_API_KEY",
            description="Optimized GPT-4 variant",
            tier="paid",
            context_window=128000,  # 128K tokens
            max_tokens=16000  # 16K max output
        ),
        ModelInfo(
            model_id="gpt-4o-mini",
            provider="openai",
            display_name="GPT-4o Mini",
            requires_key="OPENAI_API_KEY",
            description="Fast and cost-effective GPT-4o variant",
            tier="paid",
            context_window=128000,  # 128K tokens
            max_tokens=16000  # 16K max output
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
            model_id="gemini-2.5-flash",
            provider="google",
            display_name="Gemini 2.5 Flash",
            requires_key="GEMINI_API_KEY",
            description="Latest fast and efficient Gemini model",
            tier="free",
            context_window=1000000,  # 1M tokens
            max_tokens=64000  # 64K max output
        ),
        ModelInfo(
            model_id="gemini-3-1-flash",
            provider="google",
            display_name="Gemini 3.1 Flash",
            requires_key="GEMINI_API_KEY",
            description="Fast and efficient model",
            tier="free",
            context_window=1000000,  # 1M tokens
            max_tokens=64000  # 64K max output
        ),
        ModelInfo(
            model_id="gemini-3-1-pro",
            provider="google",
            display_name="Gemini 3.1 Pro",
            requires_key="GEMINI_API_KEY",
            description="Advanced reasoning capabilities (requires Pro plan)",
            tier="paid",
            context_window=1000000,  # 1M tokens
            max_tokens=65536  # 65K max output
        ),
    ],
    "groq": [
        ModelInfo(
            model_id="groq/gpt-oss-120b",
            provider="groq",
            display_name="GPT-OSS-120B",
            requires_key="GROQ_API_KEY",
            description="Open-source model on Groq infrastructure",
            tier="paid",
            context_window=131072,  # 131K tokens (estimated)
            max_tokens=8192  # 8K max output (estimated)
        ),
        ModelInfo(
            model_id="groq/llama-4-70b",
            provider="groq",
            display_name="Llama 4 70B",
            requires_key="GROQ_API_KEY",
            description="Meta's Llama 4 on Groq",
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
            model_id="deepseek/deepseek-r1:free",
            provider="openrouter",
            display_name="DeepSeek R1 (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="671B reasoning model, performance on par with OpenAI o1",
            tier="free",
            context_window=128000,  # 128K tokens
            max_tokens=32768  # 32K max output
        ),
        ModelInfo(
            model_id="deepseek/deepseek-r1-distill-qwen-32b:free",
            provider="openrouter",
            display_name="DeepSeek R1 Distill Qwen 32B (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="32B distilled reasoning model, outperforms o1-mini",
            tier="free",
            context_window=128000,  # 128K tokens
            max_tokens=16000  # 16K max output
        ),
        ModelInfo(
            model_id="deepseek/deepseek-r1-distill-qwen-14b:free",
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
            model_id="qwen/qwen3-coder:free",
            provider="openrouter",
            display_name="Qwen3 Coder 480B (Free)",
            requires_key="OPENROUTER_API_KEY",
            description="480B MoE coding model, 35B active parameters",
            tier="free",
            context_window=262144,  # 262K tokens
            max_tokens=16000  # 16K max output
        ),
        ModelInfo(
            model_id="qwen/qwen3-next-80b-a3b-instruct:free",
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