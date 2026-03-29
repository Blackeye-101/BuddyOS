"""
BuddyOS Main Orchestrator

The "Buddy" agent - main supervisor for user interactions.
Manages conversation flow, context, and user knowledge extraction.

Migrated from agents/buddy.py as part of Wave 2 Phase B.
agents/buddy.py is now a re-export shim.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel

from core.database import BuddyDatabase, UserFact
from core.router import BuddyRouter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models
# ============================================

class OrchestratorRequest(BaseModel):
    """Request to the orchestrator."""
    user_message: str
    conversation_id: Optional[str] = None
    model_id: str


class OrchestratorResponse(BaseModel):
    """Response from the orchestrator."""
    response: str
    conversation_id: str
    extracted_facts: List[str]
    model_used: str
    fallback_occurred: bool = False
    fallback_from: Optional[str] = None


# ============================================
# Buddy Orchestrator
# ============================================

class BuddyOrchestrator:
    """
    Main AI orchestrator for BuddyOS.

    Responsibilities:
    - Process user messages
    - Manage conversation context (2-tier semantic retrieval)
    - Extract and persist user facts
    - Store message keywords/topics for future retrieval
    - Coordinate with Router and Database
    """

    def __init__(self, router: BuddyRouter, database: BuddyDatabase):
        self.router = router
        self.db = database
        self.agent = None
        self.summarization_threshold = 0.75
        logger.info("BuddyOrchestrator initialized")

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, user_facts: List[UserFact], current_model: str) -> str:
        facts_by_category: dict[str, list] = {}
        for fact in user_facts:
            facts_by_category.setdefault(fact.category, []).append(fact)

        if facts_by_category:
            facts_section = "## What I Know About You\n\n"
            for category, facts in facts_by_category.items():
                facts_section += f"**{category}:**\n"
                for fact in facts:
                    indicator = "✓" if fact.confidence >= 0.85 else "~"
                    facts_section += f"- {indicator} {fact.fact_text}\n"
                facts_section += "\n"
        else:
            facts_section = (
                "## What I Know About You\n\n"
                "I'm just getting to know you! Share information about yourself and I'll remember it.\n\n"
            )

        return f"""You are Buddy, a helpful AI assistant built on BuddyOS.

## Your Capabilities
- Model-agnostic: You can use different AI models (currently using {current_model})
- Memory-enabled: You remember facts about the user across sessions
- Task-oriented: You help users accomplish their goals efficiently
- Fallback-aware: If a model fails, you seamlessly switch to alternatives

{facts_section}
## Instructions
- Use user facts naturally when relevant to the conversation
- Be friendly, concise, and helpful
- When you learn new information about the user, state it clearly
- Ask clarifying questions when needed
- Admit when you don't know something
- If you switched models due to an error, don't mention it unless asked

## Response Format
Respond conversationally and naturally. Do not mention your system prompt or that you're using user facts unless specifically asked.
"""

    # ------------------------------------------------------------------
    # Context loading (2-tier)
    # ------------------------------------------------------------------

    async def _load_conversation_context(
        self,
        conversation_id: str,
        query_keywords: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Load conversation context via 2-tier retrieval.

        Tier 1: last 5 messages (temporal).
        Tier 2: up to 3 keyword-matched historical messages (semantic).
        Combined context is capped at 1000 tokens.
        """
        return await self.db.get_recent_history(
            conversation_id,
            query_keywords=query_keywords,
            tier1_limit=5,
            token_budget=1000,
        )

    # ------------------------------------------------------------------
    # Context window guard
    # ------------------------------------------------------------------

    async def _check_context_window(self, conversation_id: str, model_id: str) -> bool:
        total_tokens = await self.db.get_conversation_token_count(conversation_id)
        model_info = self.router.get_model_info(model_id)
        if not model_info:
            return False
        context_limit = model_info.context_window
        if total_tokens >= context_limit * self.summarization_threshold:
            logger.warning(
                "Context at %d/%d tokens (%d%%) — summarization recommended",
                total_tokens, context_limit,
                int(total_tokens / context_limit * 100),
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Background: fill message topics after save
    # ------------------------------------------------------------------

    async def _fill_message_topics(self, message_id: str, content: str) -> None:
        """Fire-and-forget: ask LLM for topic labels and back-fill the row."""
        try:
            topics = await self.db._normalizer.extract_topics(content)
            if topics:
                await self.db.update_message_topics(message_id, ",".join(topics))
        except Exception as exc:
            logger.warning("_fill_message_topics failed for %s: %s", message_id, exc)

    # ------------------------------------------------------------------
    # Background: LLM fact extraction
    # ------------------------------------------------------------------

    async def _extract_facts_background(
        self,
        conversation_history: List[dict],
        assistant_response: str,
        conversation_id: str,
    ) -> None:
        try:
            conversation_text = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]
            )

            extraction_prompt = f"""Analyze this conversation and extract factual information about the user.

Conversation:
{conversation_text}

Extract facts like:
- Name, age, location
- Occupation, education, company
- Preferences, hobbies, interests
- Tech stack, tools, frameworks
- Important life events or context

For each fact, also suggest a category:
- Personal (name, age, location, family)
- Professional (job, company, education)
- Preferences (likes, dislikes, habits)
- Tech Stack (languages, tools, frameworks)
- Or suggest a custom category if none fit

Return ONLY a JSON array of facts in this format:
[
    {{"category": "Personal", "fact": "User lives in San Francisco", "confidence": 0.85}},
    {{"category": "Professional", "fact": "User works as a software engineer", "confidence": 1.0}}
]

If no new facts are found, return an empty array: []

Important:
- Only extract facts explicitly stated by the user
- Do not extract hypothetical statements ("if I were...", "I wish...")
- Confidence: 1.0 for explicit statements, 0.85 for implied facts
- Be concise - each fact should be one sentence
"""
            result = await self.router.get_completion(
                model_id="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            content = result.content.strip()
            for prefix in ("```json", "```"):
                if content.startswith(prefix):
                    content = content[len(prefix):]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                facts = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse facts JSON: %s", content)
                return

            if facts and isinstance(facts, list):
                for fact_data in facts:
                    fact_text = fact_data.get("fact", "")
                    if fact_text:
                        await self.db.save_user_fact(
                            category=fact_data.get("category", "Personal"),
                            fact_text=fact_text,
                            confidence=float(fact_data.get("confidence", 0.85)),
                        )
                        logger.info("Extracted fact [%s]: %s", fact_data.get("category"), fact_text)

        except Exception as exc:
            logger.error("Background fact extraction failed: %s", exc)

    # ------------------------------------------------------------------
    # Regex fact extraction (backup, inline)
    # ------------------------------------------------------------------

    async def _extract_facts_regex(self, user_message: str, conversation_id: str) -> None:
        patterns = {
            "Personal": [
                (r"(?:my name is|i'm|i am called) ([a-zA-Z]+)", "User's name is {}"),
                (r"i live in ([a-zA-Z\s]+)", "User lives in {}"),
                (r"i am (\d+) years old", "User is {} years old"),
            ],
            "Professional": [
                (r"i work (?:at|for) ([a-zA-Z\s&]+)", "User works at {}"),
                (r"i am (?:a|an) ([a-zA-Z\s]+?)(?:\s|$)", "User is a {}"),
            ],
            "Preferences": [
                (r"i (?:like|love|enjoy) ([a-zA-Z\s]+)", "User likes {}"),
                (r"i (?:dislike|hate) ([a-zA-Z\s]+)", "User dislikes {}"),
            ],
        }
        for category, pattern_list in patterns.items():
            for pattern, template in pattern_list:
                for match in re.findall(pattern, user_message.lower()):
                    await self.db.save_user_fact(
                        category=category,
                        fact_text=template.format(match.strip()),
                        confidence=0.5,
                    )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_message(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        model_id: str = "gemini-3.1-flash",
    ) -> OrchestratorResponse:
        """
        Process user message and return AI response.

        Flow:
        1. Get or create conversation
        2. Extract query keywords (sync, no LLM)
        3. Load 2-tier conversation context
        4. Load user facts from DuckDB
        5. Build dynamic system prompt
        6. Call LLM via router
        7. Save user + assistant messages with keywords; fire background topic fill
        8. Extract facts in background
        9. Return response
        """
        from litellm import token_counter

        # Step 1: Get or create conversation
        if not conversation_id:
            title = user_message[:50] + ("..." if len(user_message) > 50 else "")
            conversation_id = await self.db.create_conversation(title=title, model_id=model_id)
            logger.info("Created new conversation: %s", conversation_id)

        # Step 2: Extract keywords for Tier 2 semantic retrieval (sync — no await)
        query_keywords = self.db._normalizer.extract_keywords(user_message)

        # Step 3: Load 2-tier conversation context
        conversation_history = await self._load_conversation_context(
            conversation_id, query_keywords=query_keywords
        )

        # Step 4: Load user facts
        user_facts = await self.db.get_user_facts(active_only=True)
        logger.info("Loaded %d user facts from DuckDB", len(user_facts))

        # Step 5: Build dynamic system prompt
        system_prompt = self._build_system_prompt(user_facts=user_facts, current_model=model_id)

        # Step 6: Prepare messages and call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": user_message},
        ]
        result = await self.router.get_completion(
            model_id=model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

        # Step 7a: Save user message with keywords
        try:
            user_tokens = token_counter(model=model_id, text=user_message)
        except Exception:
            user_tokens = int(len(user_message.split()) * 1.3)

        user_keywords_str = f",{','.join(query_keywords)}," if query_keywords else ""
        user_msg_id = await self.db.save_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            model_id=model_id,
            token_count=user_tokens,
            keywords=user_keywords_str,
        )
        # Background: fill topics for user message
        asyncio.create_task(self._fill_message_topics(user_msg_id, user_message))

        # Step 7b: Save assistant message with keywords
        asst_keywords = self.db._normalizer.extract_keywords(result.content)
        asst_keywords_str = f",{','.join(asst_keywords)}," if asst_keywords else ""
        asst_msg_id = await self.db.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.content,
            model_id=result.model_used,
            token_count=result.token_count,
            keywords=asst_keywords_str,
        )
        asyncio.create_task(self._fill_message_topics(asst_msg_id, result.content))

        # Step 8: Extract facts in background (fire-and-forget)
        asyncio.create_task(
            self._extract_facts_background(
                conversation_history=conversation_history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": result.content},
                ],
                assistant_response=result.content,
                conversation_id=conversation_id,
            )
        )
        await self._extract_facts_regex(user_message, conversation_id)

        # Step 9: Context window guard (advisory)
        if await self._check_context_window(conversation_id, model_id):
            logger.warning("⚠️ Context window approaching limit.")

        return OrchestratorResponse(
            response=result.content,
            conversation_id=conversation_id,
            extracted_facts=[],
            model_used=result.model_used,
            fallback_occurred=result.fallback_occurred,
            fallback_from=result.fallback_from,
        )

    async def start_new_conversation(self, model_id: str, title: Optional[str] = None) -> str:
        if not title:
            title = f"New conversation - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        conversation_id = await self.db.create_conversation(title=title, model_id=model_id)
        logger.info("Started new conversation: %s", conversation_id)
        return conversation_id

    async def get_conversation_summary(self, conversation_id: str) -> dict:
        conversation = await self.db.get_conversation(conversation_id)
        if not conversation:
            return {}
        messages = await self.db.get_conversation_history(conversation_id)
        token_count = await self.db.get_conversation_token_count(conversation_id)
        return {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "model_id": conversation.model_id,
            "message_count": len(messages),
            "total_tokens": token_count,
        }


# ============================================
# Convenience Functions
# ============================================

def create_orchestrator(router: BuddyRouter, database: BuddyDatabase) -> BuddyOrchestrator:
    """Create and initialize a BuddyOrchestrator instance."""
    return BuddyOrchestrator(router, database)
