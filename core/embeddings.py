"""
BuddyOS Local Embedding Utility

Generates 384-dimensional text embeddings locally via FastEmbed (ONNX Runtime).
No PyTorch required. Model downloads ~25 MB on first use.

Model : BAAI/bge-small-en-v1.5
Output: 384-float list, L2-normalised (cosine-similarity ready)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
EMBEDDING_DIM = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Store model cache inside the project to avoid Windows 8.3 short-path issues
# with the default %TEMP% directory (e.g. RISHAB~1).
_CACHE_DIR = str(Path(__file__).resolve().parent.parent / "data" / "fastembed_cache")

# ── Lazy singleton ──────────────────────────────────────────────────────────
_model = None  # Loaded exactly once on first call


def _get_model():
    """Load the FastEmbed model (sync). Called inside a worker thread."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        os.makedirs(_CACHE_DIR, exist_ok=True)
        logger.info(
            "Loading local embedding model '%s' — first run downloads ~66 MB...",
            MODEL_NAME,
        )
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=_CACHE_DIR)
        logger.info("FastEmbed model loaded successfully.")
    return _model


# ── Public API ──────────────────────────────────────────────────────────────

def embed_sync(text: str) -> list[float]:
    """
    Embed *text* synchronously.
    Returns a 384-float list normalised to unit length.
    Intended only for callers already inside a worker thread.
    """
    model = _get_model()
    # FastEmbed returns a generator; consume the first (and only) result
    vectors = list(model.embed([text]))
    return vectors[0].tolist()


async def generate_embedding(text: str) -> list[float]:
    """
    Embed *text* asynchronously (CPU-bound encode runs in a thread pool).

    Args:
        text: Text to embed.

    Returns:
        384-float list ready for DuckDB FLOAT[384] storage.
    """
    return await asyncio.to_thread(embed_sync, text)


async def generate_embedding_safe(text: str) -> Optional[list[float]]:
    """
    Same as *generate_embedding* but catches all exceptions and returns None
    on failure so callers do not need to wrap it themselves.
    """
    try:
        return await generate_embedding(text)
    except Exception as exc:
        logger.warning("Embedding generation failed: %s", exc)
        return None
