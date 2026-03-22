"""
BuddyOS Main Orchestrator

The "Buddy" agent - main supervisor for user interactions.
Manages conversation flow, context, and user knowledge extraction.
"""

import asyncio
import logging
import re
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from core.deps import BuddyDeps
from core.database import BuddyDatabase, UserFact
from core.router import BuddyRouter, CompletionResult
from core.models import ModelInfo


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
    - Manage conversation context
    - Extract and persist user facts
    - Coordinate with Router and Database
    - Manage context window with summarization
    """
    
    def __init__(self, router: BuddyRouter, database: BuddyDatabase):
        """
        Initialize orchestrator with router and database.
        
        Args:
            router: BuddyRouter instance for model access
            database: BuddyDatabase instance for persistence
        """
        self.router = router
        self.db = database
        
        # Create PydanticAI agent (will be configured per request)
        self.agent = None
        
        # Context management
        self.max_history_messages = 20  # Maximum messages to include in context
        self.summarization_threshold = 0.75  # Trigger summarization at 75% of context
        
        logger.info("BuddyOrchestrator initialized")
    
    def _build_system_prompt(self, user_facts: List[UserFact], current_model: str) -> str:
        """
        Build dynamic system prompt with user facts.
        
        Args:
            user_facts: List of user facts from DuckDB
            current_model: Model identifier being used
            
        Returns:
            System prompt string
        """
        # Group facts by category
        facts_by_category = {}
        for fact in user_facts:
            category = fact.category
            if category not in facts_by_category:
                facts_by_category[category] = []
            facts_by_category[category].append(fact)
        
        # Build facts section
        facts_section = ""
        if facts_by_category:
            facts_section = "## What I Know About You\n\n"
            for category, facts in facts_by_category.items():
                facts_section += f"**{category}:**\n"
                for fact in facts:
                    confidence_indicator = "✓" if fact.confidence >= 0.85 else "~"
                    facts_section += f"- {confidence_indicator} {fact.fact_text}\n"
                facts_section += "\n"
        else:
            facts_section = "## What I Know About You\n\nI'm just getting to know you! Share information about yourself and I'll remember it.\n\n"
        
        # Build system prompt
        system_prompt = f"""You are Buddy, a helpful AI assistant built on BuddyOS.

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
        
        return system_prompt
    
    async def _load_conversation_context(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[dict]:
        """
        Load recent messages as context.
        
        Args:
            conversation_id: UUID of conversation
            limit: Maximum number of messages to load
            
        Returns:
            List of message dicts for PydanticAI
        """
        # Get recent history from database
        history = await self.db.get_recent_history(conversation_id, limit=limit)
        
        return history
    
    async def _check_context_window(
        self,
        conversation_id: str,
        model_id: str
    ) -> bool:
        """
        Check if conversation needs summarization.
        
        Args:
            conversation_id: UUID of conversation
            model_id: Current model being used
            
        Returns:
            True if summarization needed, False otherwise
        """
        # Get total token count from database
        total_tokens = await self.db.get_conversation_token_count(conversation_id)
        
        # Get model context window
        model_info = self.router.get_model_info(model_id)
        if not model_info:
            return False
        
        context_limit = model_info.context_window
        
        # Check if we're at 75% threshold
        if total_tokens >= (context_limit * self.summarization_threshold):
            logger.warning(
                f"Context at {total_tokens}/{context_limit} tokens "
                f"({int((total_tokens/context_limit)*100)}%) - summarization recommended"
            )
            return True
        
        return False
    
    async def _extract_facts_background(
        self,
        conversation_history: List[dict],
        assistant_response: str,
        conversation_id: str
    ):
        """
        Extract user facts in background (fire-and-forget).
        
        Uses Gemini 3.1 Flash for fast, free extraction.
        
        Args:
            conversation_history: Recent conversation messages
            assistant_response: Latest assistant response
            conversation_id: Source conversation ID
        """
        try:
            # Build extraction prompt
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])
            
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
            
            # Use Gemini 3.1 Flash for extraction (free, fast)
            messages = [{"role": "user", "content": extraction_prompt}]
            
            result = await self.router.get_completion(
                model_id="gemini-3.1-flash",
                messages=messages,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            
            # Clean response (remove markdown code blocks if present)
            content = result.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            # Parse facts
            try:
                facts = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse facts JSON: {content}")
                return
            
            # Save facts to DuckDB
            if facts and isinstance(facts, list):
                for fact_data in facts:
                    category = fact_data.get("category", "Personal")
                    fact_text = fact_data.get("fact", "")
                    confidence = fact_data.get("confidence", 0.85)
                    
                    if fact_text:
                        await self.db.save_user_fact(
                            category=category,
                            fact_text=fact_text,
                            confidence=float(confidence)
                        )
                        logger.info(f"Extracted fact [{category}]: {fact_text} (confidence: {confidence})")
            
        except Exception as e:
            # Log error but don't block conversation
            logger.error(f"Background fact extraction failed: {e}")
    
    async def _extract_facts_regex(self, user_message: str, conversation_id: str):
        """
        Simple regex-based fact extraction (backup method).
        
        Args:
            user_message: User's message
            conversation_id: Source conversation ID
        """
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
                matches = re.findall(pattern, user_message.lower())
                for match in matches:
                    fact_text = template.format(match.strip())
                    await self.db.save_user_fact(
                        category=category,
                        fact_text=fact_text,
                        confidence=0.5  # Lower confidence for regex
                    )
                    logger.info(f"Regex extracted fact [{category}]: {fact_text} (confidence: 0.5)")
    
    async def process_message(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        model_id: str = "gemini-3.1-flash"
    ) -> OrchestratorResponse:
        """
        Process user message and return AI response.
        
        Flow:
        1. Get or create conversation
        2. Load conversation history
        3. Load user facts from DuckDB
        4. Build dynamic system prompt
        5. Call LLM via router
        6. Extract facts in background
        7. Save messages to database
        8. Return response
        
        Args:
            user_message: User's input message
            conversation_id: Optional conversation ID (creates new if None)
            model_id: Model to use for completion
            
        Returns:
            OrchestratorResponse with reply and metadata
        """
        # Step 1: Get or create conversation
        if not conversation_id:
            # Generate title from first message (truncated)
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message
            conversation_id = await self.db.create_conversation(
                title=title,
                model_id=model_id
            )
            logger.info(f"Created new conversation: {conversation_id}")
        
        # Step 2: Load conversation history
        conversation_history = await self._load_conversation_context(
            conversation_id,
            limit=self.max_history_messages
        )
        
        # Step 3: Load user facts
        user_facts = await self.db.get_user_facts(active_only=True)
        logger.info(f"Loaded {len(user_facts)} user facts from DuckDB")
        
        # Step 4: Build dynamic system prompt
        system_prompt = self._build_system_prompt(
            user_facts=user_facts,
            current_model=model_id
        )
        
        # Step 5: Prepare messages for LLM
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Step 6: Get completion from router
        result = await self.router.get_completion(
            model_id=model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Step 7: Save user message to database
        from litellm import token_counter
        
        try:
            user_tokens = token_counter(model=model_id, text=user_message)
        except Exception:
            user_tokens = int(len(user_message.split()) * 1.3)
        
        await self.db.save_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            model_id=model_id,
            token_count=user_tokens
        )
        
        # Step 8: Save assistant response to database
        await self.db.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.content,
            model_id=result.model_used,
            token_count=result.token_count
        )
        
        # Step 9: Extract facts in background (fire-and-forget)
        # Use LLM-based extraction for better accuracy
        asyncio.create_task(
            self._extract_facts_background(
                conversation_history=conversation_history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": result.content}
                ],
                assistant_response=result.content,
                conversation_id=conversation_id
            )
        )
        
        # Also try regex extraction (fast, runs inline)
        await self._extract_facts_regex(user_message, conversation_id)
        
        # Step 10: Check if summarization is needed
        needs_summarization = await self._check_context_window(
            conversation_id=conversation_id,
            model_id=model_id
        )
        
        if needs_summarization:
            logger.warning(
                "⚠️ Context window approaching limit. "
                "Consider implementing summarization tool in future iteration."
            )
        
        # Step 11: Return response
        return OrchestratorResponse(
            response=result.content,
            conversation_id=conversation_id,
            extracted_facts=[],  # Facts extracted in background
            model_used=result.model_used,
            fallback_occurred=result.fallback_occurred,
            fallback_from=result.fallback_from
        )
    
    async def start_new_conversation(self, model_id: str, title: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            model_id: Model to use for the conversation
            title: Optional conversation title
            
        Returns:
            conversation_id: UUID of created conversation
        """
        if not title:
            title = f"New conversation - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        
        conversation_id = await self.db.create_conversation(
            title=title,
            model_id=model_id
        )
        
        logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id
    
    async def get_conversation_summary(self, conversation_id: str) -> dict:
        """
        Get summary of a conversation.
        
        Args:
            conversation_id: UUID of conversation
            
        Returns:
            Dict with conversation metadata and statistics
        """
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
            "total_tokens": token_count
        }


# ============================================
# Convenience Functions
# ============================================

def create_orchestrator(router: BuddyRouter, database: BuddyDatabase) -> BuddyOrchestrator:
    """
    Create and initialize a BuddyOrchestrator instance.
    
    Args:
        router: BuddyRouter instance
        database: BuddyDatabase instance
        
    Returns:
        Initialized BuddyOrchestrator
    """
    return BuddyOrchestrator(router, database)