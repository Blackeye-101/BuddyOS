"""
BuddyOS Fact Normalization Utility

Converts raw fact strings into deterministic snake_case keys (slugs)
used for deduplication in the DuckDB facts table.

Strategy:
  1. Try regex rules for known fact patterns (fast, deterministic).
  2. Fall back to a live LiteLLM call for unrecognised patterns.
  3. On any LLM error, return a safe fallback: {category_slug}_{uuid8}.
"""

import re
import logging
from typing import Optional
from uuid import uuid4

import litellm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords for keyword extraction
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "that", "this", "with", "are", "was", "were",
    "have", "has", "had", "not", "but", "from", "they", "she", "him",
    "his", "her", "its", "our", "your", "can", "will", "just", "out",
    "about", "into", "than", "then", "some", "who", "what", "when",
    "how", "also", "been", "more", "like", "use", "you", "all", "any",
    "one", "two", "get", "let", "set", "run", "new", "see", "way",
    "did", "make", "yes", "okay",
})

# ---------------------------------------------------------------------------
# Regex rules
# Each entry: (compiled_pattern, key_template, is_additive)
#   is_additive=False → key_template is the full key (singleton fact)
#   is_additive=True  → key_template contains {0}, filled by slugified capture
# ---------------------------------------------------------------------------

_NON_ROLE_ADJECTIVES = r"(?!(?:interested|happy|ready|sure|aware|not|still|already)\b)"

REGEX_RULES: list[tuple[re.Pattern, str, bool]] = [
    # Personal singletons
    (
        re.compile(r"user(?:'s)? name is (\w+)", re.IGNORECASE),
        "personal_name",
        False,
    ),
    (
        re.compile(r"user (?:lives|is (?:located|based)) in ([\w\s]+)", re.IGNORECASE),
        "personal_location",
        False,
    ),
    (
        re.compile(r"user is (\d+) years old", re.IGNORECASE),
        "personal_age",
        False,
    ),
    # Professional singletons
    (
        re.compile(r"user works (?:at|for|with) ([\w\s]+)", re.IGNORECASE),
        "professional_employer",
        False,
    ),
    (
        re.compile(
            rf"user (?:is|works as) an? {_NON_ROLE_ADJECTIVES}([\w\s]+)",
            re.IGNORECASE,
        ),
        "professional_role",
        False,
    ),
    # Additive preferences / tech / skills
    (
        re.compile(r"user (?:prefers|likes|loves|enjoys) ([\w\s]+)", re.IGNORECASE),
        "preference_{0}",
        True,
    ),
    (
        re.compile(r"user uses ([\w\s]+)", re.IGNORECASE),
        "tech_stack_{0}",
        True,
    ),
    (
        re.compile(r"user knows ([\w\s]+)", re.IGNORECASE),
        "skill_{0}",
        True,
    ),
]

SINGLETON_KEYS: frozenset[str] = frozenset(
    {
        "personal_name",
        "personal_location",
        "personal_age",
        "professional_employer",
        "professional_role",
    }
)


class FactNormalizer:
    """Converts fact text into deterministic snake_case keys for deduplication."""

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"[^a-z0-9_]", "", text)
        return text.strip("_")[:80]

    async def normalize(self, category: str, fact_text: str) -> str:
        """
        Return a snake_case key for the given fact.

        Tries regex rules first; falls back to LLM for unrecognised patterns.
        """
        lower = fact_text.strip().lower()
        for pattern, template, is_additive in REGEX_RULES:
            m = pattern.search(lower)
            if m:
                if is_additive:
                    return self._slugify(template.format(self._slugify(m.group(1))))
                return template

        return await self._llm_normalize(category, fact_text)

    async def _llm_normalize(self, category: str, fact_text: str) -> str:
        """Ask the LLM to produce a snake_case key for an unrecognised fact."""
        prompt = (
            f"Return ONLY a snake_case key (max 6 words, lowercase, underscores only) "
            f"that identifies the TYPE of information — not the value itself — "
            f'for this fact: "{fact_text}" in category "{category}". '
            f"Examples: personal_location, preference_music_genre, tech_stack_language. "
            f"Output the key alone on a single line."
        )
        try:
            response = await litellm.acompletion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip().splitlines()[0]
            return self._slugify(raw)
        except Exception as exc:
            logger.warning("FactNormalizer LLM fallback failed: %s", exc)
            return f"{self._slugify(category)}_{uuid4().hex[:8]}"

    def is_contradiction(
        self,
        fact_key: str,
        old_fact_text: str,
        new_fact_text: str,
    ) -> bool:
        """Return True if the new fact contradicts an existing singleton fact."""
        return (
            fact_key in SINGLETON_KEYS
            and old_fact_text.strip().lower() != new_fact_text.strip().lower()
        )

    # ------------------------------------------------------------------
    # Semantic history helpers
    # ------------------------------------------------------------------

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract meaningful keywords from text synchronously.

        Returns up to 10 deduplicated lowercase tokens, filtered against
        STOPWORDS. Used to populate the `keywords` column on messages.
        """
        tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text.lower())
        seen: dict[str, None] = {}
        for tok in tokens:
            if tok not in STOPWORDS and tok not in seen:
                seen[tok] = None
            if len(seen) == 10:
                break
        return list(seen.keys())

    async def extract_topics(self, text: str) -> list[str]:
        """
        Use the LLM to derive 1-3 high-level topic labels (snake_case).

        Falls back to the first two keywords on any error.
        """
        prompt = (
            "Return 1-3 comma-separated topic labels (lowercase snake_case, "
            "e.g. python_programming, travel_plans) that best describe the "
            f"subject of this message: \"{text}\". "
            "Output labels only, nothing else."
        )
        try:
            response = await litellm.acompletion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip().splitlines()[0]
            return [self._slugify(t) for t in raw.split(",") if t.strip()][:3]
        except Exception as exc:
            logger.warning("FactNormalizer.extract_topics LLM failed: %s", exc)
            return [self._slugify(kw) for kw in self.extract_keywords(text)[:2]]

    async def extract_metadata(self, text: str) -> tuple[list[str], list[str]]:
        """
        Return (keywords, topics) for a message.

        keywords — fast sync extraction, populated before save_message returns.
        topics   — async LLM extraction, populated in a background task.
        """
        keywords = self.extract_keywords(text)
        topics = await self.extract_topics(text)
        return keywords, topics
