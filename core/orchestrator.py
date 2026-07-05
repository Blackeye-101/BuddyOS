import asyncio
import json
import logging
import re
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from core.database import BuddyDatabase, UserFact
from core.embeddings import generate_embedding_safe
from core.router import BuddyRouter
from core.tools import ToolRegistry

logger = logging.getLogger(__name__)

class OrchestratorRequest(BaseModel):
    user_message: str
    conversation_id: Optional[str] = None
    model_id: str


class OrchestratorResponse(BaseModel):
    response: str
    conversation_id: str
    extracted_facts: List[str]
    model_used: str
    fallback_occurred: bool = False
    fallback_from: Optional[str] = None


class FactExtractionItem(BaseModel):
    category: str = Field(description="The general category of the fact, e.g., 'Personal', 'Preferences', 'Pets', etc.")
    fact_text: str = Field(description="The actual fact text")

class FactExtractionSchema(BaseModel):
    add_facts: List[FactExtractionItem] = Field(default_factory=list, description="New facts discovered in the user's message")
    deactivate_fact_ids: List[str] = Field(default_factory=list, description="UUIDs of existing facts that are explicitly contradicted and should be removed")


class BaseAgent:
    def __init__(self, agent_name: str, router: BuddyRouter, tool_registry: ToolRegistry):
        self.agent_name = agent_name
        self.router = router
        self.tool_registry = tool_registry

    async def run(self, messages: list, model_id: str, temperature: float = 0.7, max_steps: int = 5) -> tuple[str, str, int, bool, Optional[str]]:
        tools = self.tool_registry.get_tools_for_agent(self.agent_name)
        result_content = ""
        result_model_used = model_id
        result_token_count = 0
        fallback_occurred = False
        fallback_from = None
        step_count = 0
        
        while step_count < max_steps:
            kwargs = {}
            if tools and step_count < max_steps - 1:
                kwargs["tools"] = tools
                
            result = await self.router.get_completion(
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
                **kwargs
            )
            
            result_model_used = result.model_used
            result_token_count += result.token_count
            if result.fallback_occurred:
                fallback_occurred = True
                fallback_from = result.fallback_from
                
            if result.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": result.content or "",  # Add empty string if None
                    "tool_calls": result.tool_calls
                })
                
                for tool_call in result.tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    logger.info(f"{self.agent_name} executing tool: {function_name}")
                    
                    tool_result_str = self.tool_registry.execute_tool(function_name, function_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result_str
                    })
            else:
                result_content = result.content
                break
                
            step_count += 1

        # GPT-5 and other models might return `content=None` when `tool_calls` are present. 
        # In step max_steps, we strip tools, forcing a text return, but if that still returns empty:
        if not result_content and result and result.content:
            result_content = result.content
            
        return result_content or "", result_model_used, result_token_count, fallback_occurred, fallback_from


class BuddyOrchestrator:
    """
    Main AI orchestrator acting as a Supervisor.
    Routes queries to BuddyAgent or ResearcherAgent.
    """

    def __init__(self, router: BuddyRouter, database: BuddyDatabase):
        from core.fact_utils import FactNormalizer
        self.router = router
        self.db = database
        self.summarization_threshold = 0.75
        self._background_tasks: set = set()
        self.db._normalizer = FactNormalizer(router=router)
        self.tool_registry = ToolRegistry(db=database, router=router, supervisor=self)
        
        self.buddy_agent = BaseAgent("buddy", router, self.tool_registry)
        self.researcher_agent = BaseAgent("researcher", router, self.tool_registry)
        
        logger.info("Supervisor Orchestrator initialized")

    def run_agent_sync(self, target_agent: str, prompt: str) -> str:
        """Synchronous wrapper to run an agent from a tool."""
        logger.info(f"Handoff from tool to agent: {target_agent}")
        agent = getattr(self, f"{target_agent}_agent", None)
        if not agent:
            return f"Agent {target_agent} not found."
            
        messages = [{"role": "user", "content": prompt}]
        # Try to safely execute the async run method
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                agent.run(messages, "gemini/gemini-2.5-flash"), loop
            )
            result = future.result(timeout=60)
            return result[0]
        except RuntimeError:
            result = asyncio.run(agent.run(messages, "gemini/gemini-2.5-flash"))
            return result[0]

    def _launch_background_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def wait_background_tasks(self) -> None:
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

    def _build_buddy_prompt(self, user_facts: List[UserFact], current_model: str) -> str:
        facts_by_category: dict[str, list] = {}
        for fact in user_facts:
            facts_by_category.setdefault(fact.category, []).append(fact)

        facts_section = "## What I Know About You\n\n"
        if facts_by_category:
            for category, facts in facts_by_category.items():
                facts_section += f"**{category}:**\n"
                for fact in facts:
                    indicator = "✓" if fact.confidence >= 0.85 else "~"
                    facts_section += f"- {indicator} {fact.fact_text}\n"
                facts_section += "\n"
        else:
            facts_section += "I'm just getting to know you! Share information about yourself and I'll remember it.\n\n"

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
- Ask clarifying questions when needed
- You have the ability to search the web using duckduckgo. Use the web_search tool for real-time information.
- IMPORTANT DELEGATION: If the user asks for rigorously peer-reviewed academic papers or deeply academic topics, you MUST use the `delegate_to_researcher_for_academic` tool to hand off the question to the Researcher agent.
"""

    def _build_researcher_prompt(self) -> str:
        return """You are an Academic Researcher Agent in BuddyOS.

## Instructions & Guardrails
- **Zero Hallucination Strictness:** NEVER invent or guess papers, authors, publication years, or DOIs. If no papers are found via tools, state that explicitly.
- **Strict Tool Reliance:** You must rely entirely on `arxiv_search` for academic facts.
- **Delegation:** Use delegate_to_buddy_for_web_search ONLY when the user asks for general web information, recent news, stock prices, or events not fit for ArXiv.
- **Mandatory Inline Citations:** Every academic claim must be cited with Title, Primary Author, Year, and a link.
- **Objectivity & Limitations:** Maintain an objective, academic tone.
"""

    async def _load_conversation_context(self, conversation_id: str, query_keywords: Optional[List[str]] = None) -> List[dict]:
        return await self.db.get_recent_history(
            conversation_id, query_keywords=query_keywords, tier1_limit=5, token_budget=1000
        )

    async def _summarize_oldest_context(self, conversation_id: str, model_id: str) -> None:
        """
        Summarizes the oldest 50% of the messages to save tokens.
        """
        messages = await self.db.get_conversation_history(conversation_id)
        if len(messages) <= 4:
            return

        oldest_messages = await self.db.get_oldest_messages(conversation_id, limit=len(messages) // 2)
        if not oldest_messages:
            return
            
        fast_model = self.router.get_fast_model_for_provider(model_id)
        
        text_to_summarize = "\n".join([f"{msg.role}: {msg.content}" for msg in oldest_messages if msg.role != "system"])
        
        prompt = f"""Summarize the following conversation history chronologically. 
        Focus on the main topics, facts exchanged, and key decisions.
        Keep it dense and factual.
        
        History:
        {text_to_summarize}
        """
        
        try:
            result = await self.router.get_completion(
                model_id=fast_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            summary = result.content
            
            message_ids_to_delete = [msg.id for msg in oldest_messages if msg.role != "system"]
            await self.db.delete_messages(message_ids_to_delete)
            
            from litellm import token_counter
            try:
                summary_tokens = token_counter(model=fast_model, text=summary)
            except Exception:
                summary_tokens = int(len((summary or "").split()) * 1.3)
                
            await self.db.save_message(
                conversation_id=conversation_id,
                role="system",
                content=f"[SYSTEM MEMORY: Condensed history of earlier conversation]\n{summary}",
                model_id=fast_model,
                token_count=summary_tokens
            )
            
            logger.info(f"Summarized {len(message_ids_to_delete)} messages into {summary_tokens} tokens using {fast_model}")
            
        except Exception as e:
            logger.error(f"Failed to summarize context: {e}")

    async def _check_context_window(self, conversation_id: str, model_id: str) -> bool:
        total_tokens = await self.db.get_conversation_token_count(conversation_id)
        model_info = self.router.get_model_info(model_id)
        if not model_info or not model_info.context_window: 
            return False
            
        if total_tokens >= model_info.context_window * self.summarization_threshold:
            logger.warning("Context approaching limit.")
            self._launch_background_task(
                self._summarize_oldest_context(conversation_id, model_id)
            )
            return True
        return False

    async def _generate_conversation_title(self, conversation_id: str, model_id: str) -> None:
        """
        Generate a short human-readable title for the conversation and persist it.
        Triggered as a background task after the 2nd full exchange (4th message)
        so there is enough context for a meaningful title.

        Uses whatever model the user is currently on; the Router's built-in
        fallback chain ensures a free alternative is tried if the primary model
        rejects or rate-limits the request.
        """
        try:
            history = await self.db.get_conversation_history(conversation_id, limit=4)
            if not history:
                return

            # Build a compact transcript (cap each turn at 200 chars to save tokens)
            transcript = "\n".join(
                f"{msg.role.capitalize()}: {msg.content[:200]}"
                for msg in history
            )

            # --- Primary path: ask the active model for a short title -------
            title = ""
            try:
                result = await self.router.get_completion(
                    model_id=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Write a 3-5 word title that concisely captures the topic "
                                "of the following conversation. Reply with ONLY the title, "
                                "no quotes, no punctuation at the end.\n\n"
                                f"{transcript}"
                            ),
                        }
                    ],
                    max_tokens=20,
                    temperature=0.3,
                )
                title = (result.content or "").strip()
            except Exception as llm_exc:
                # LLM unavailable (rate-limit, quota, etc.) — derive a
                # heuristic title from the first user message instead so the
                # sidebar always shows something meaningful.
                logger.warning(
                    "LLM title generation failed (%s), using heuristic fallback.", llm_exc
                )
                first_user_msg = next(
                    (msg.content for msg in history if msg.role == "user"), ""
                )
                title = first_user_msg[:45].strip()
                if len(first_user_msg) > 45:
                    title += "…"

            if title:
                await self.db.update_conversation_summary(conversation_id, title)
                logger.info(
                    "Conversation title set: '%s' for %s", title, conversation_id
                )

        except Exception as exc:
            logger.warning("Conversation title generation failed: %s", exc)

    async def _fill_message_topics(self, message_id: str, content: str, model_id: str) -> None:
        try:
            topics = await self.db._normalizer.extract_topics(content, model=model_id)
            if topics:
                await self.db.update_message_topics(message_id, ",".join(topics))
        except Exception:
            pass

    async def _extract_facts_background(self, user_message: str, model_id: str):
        try:
            # 1. Contextual Retrieval
            query_embedding = await generate_embedding_safe(user_message)
            if query_embedding is None or not self.db._vss_available:
                return

            existing_facts = await self.db.get_relevant_facts(query_embedding, limit=10)
            
            # Format existing facts for the prompt
            facts_context = ""
            if existing_facts:
                facts_context = "### Existing Facts in Database:\n"
                for fact in existing_facts:
                    facts_context += f"- ID: {fact.id} | Category: {fact.category} | Fact: {fact.fact_text}\n"
            else:
                facts_context = "No existing relevant facts in database.\n"

            # 2. Dynamic Model Selection
            fast_model = self.router.get_fast_model_for_provider(model_id)

            # 3. LLM Evaluation
            prompt = f"""You are BuddyOS's Memory Extractor.
Analyze the user's latest statement and extract any new, meaningful facts, preferences, or personal details to remember. 
If the user's new statement explicitly contradicts and invalidates any of the 'Existing Facts', output the UUID of that existing fact in 'deactivate_fact_ids'.

CRITICAL RULES FOR DEACTIVATING FACTS:
1. DO NOT deactivate past historical events just because current circumstances changed (e.g., past pets or jobs remain true historical facts, even if the user gets a new pet or job).
2. DO NOT deactivate future plans unless the user EXPLICITLY cancels or abandons them. (e.g., getting a Cat today does not cancel a plan to get a Dog in the future).
3. Only deactivate a fact if the new statement represents a direct, mutually exclusive contradiction of a present-state fact (e.g., "I actually hate apples" contradicts "I love apples").

Otherwise, if elements are entirely new, add them as 'add_facts'. Treat implicit overlaps flexibly.

### User Statement:
"{user_message}"

{facts_context}
"""

            result = await self.router.get_completion(
                model_id=fast_model,
                messages=[{"role": "user", "content": prompt}],
                response_format=FactExtractionSchema,
                max_tokens=1000
            )

            # 4. Execution
            if not result.content:
                return

            # LiteLLM handles Pydantic outputs, so result.content is a JSON string
            extracted_data = json.loads(result.content)
            
            for deactivated_id in extracted_data.get("deactivate_fact_ids", []):
                logger.info(f"Deactivating contradicted fact ID: {deactivated_id}")
                await self.db.deactivate_fact(deactivated_id)
                
            for new_fact in extracted_data.get("add_facts", []):
                logger.info(f"Adding new user fact via LLM: {new_fact['fact_text']}")
                await self.db.save_user_fact(
                    category=new_fact["category"],
                    fact_text=new_fact["fact_text"],
                    confidence=0.5,
                    model_id=fast_model
                )

        except Exception as e:
            logger.error(f"Background Fact Extraction failed: {e}")

    async def _extract_facts_regex(self, user_message: str, conversation_id: str, model_id: Optional[str] = None) -> None:
        patterns = {
            "Personal": [(r"i live in ([a-zA-Z\s]+)", "User lives in {}")],
        }
        for category, pattern_list in patterns.items():
            for pattern, template in pattern_list:
                for match in re.findall(pattern, user_message.lower()):
                    await self.db.save_user_fact(category=category, fact_text=template.format(match.strip()), confidence=0.5, model_id=model_id)

    async def _determine_intent(self, user_message: str, model_id: str) -> str:
        """Evaluate user intent to route to researcher, finance, or buddy using a cascading strategy."""
        um_lower = user_message.lower()
        
        # 1. Fast Path Keyword Intercepts
        if any(kw in um_lower for kw in ["arxiv", "research paper", "academic paper", "academic research"]):
            return "researcher"
        if any(kw in um_lower for kw in ["stock", "ticker", "portfolio", "nse", "bse", "market cap", "valuation"]):
            return "finance"
            
        # 2. LLM Classification Safety Net
        prompt = f"""Classify the following query into exactly ONE of the following categories:
- 'academic': Focuses on theoretical research, whitepapers, university studies, or academic journals (Even if the subject is economics).
- 'finance': Focuses on live stock markets, trading, live company analysis, portfolios, or corporate news.
- 'general': Anything else (chat, coding, general knowledge).

Query: "{user_message}"
Respond with ONLY the category word ('academic', 'finance', or 'general')."""
        res = await self.router.get_completion(model_id=model_id, messages=[{"role": "user", "content": prompt}], max_tokens=10)
        content = (res.content or "general").strip().lower()
        if "academic" in content:
            return "researcher"
        elif "finance" in content:
            return "finance"
        return "buddy"

    async def process_message(
        self, user_message: str, conversation_id: Optional[str] = None, model_id: str = "gemini-3.1-flash"
    ) -> OrchestratorResponse:
        from litellm import token_counter

        if not conversation_id:
            title = user_message[:50] + ("..." if len(user_message) > 50 else "")
            conversation_id = await self.db.create_conversation(title=title, model_id=model_id)
            logger.info("Created new conversation: %s", conversation_id)

        query_keywords = self.db._normalizer.extract_keywords(user_message)
        conversation_history = await self._load_conversation_context(conversation_id, query_keywords=query_keywords)

        # 1. Routing
        intent = await self._determine_intent(user_message, model_id)
        logger.info(f"Supervisor routed query to: {intent}")

        # 2. Context / Agents
        is_finance_workflow = False
        if intent == "researcher":
            system_prompt = self._build_researcher_prompt()
            agent = self.researcher_agent
            messages = [{"role": "system", "content": system_prompt}] + conversation_history + [{"role": "user", "content": user_message}]
        elif intent == "finance":
            from agents.finance import FinanceWorkflow
            is_finance_workflow = True
            finance_wf = FinanceWorkflow(self.router, self.tool_registry)
        else:
            query_embedding = await generate_embedding_safe(user_message)
            user_facts = []
            if query_embedding is not None and self.db._vss_available:
                user_facts = await self.db.get_relevant_facts(query_embedding, limit=10)
            else:
                user_facts = await self.db.get_user_facts(active_only=True)
            
            # Personal RAG Injection
            document_context = ""
            # Simple heuristic to trigger document search: mentioning "document", "file", "csv", "pdf", etc.
            if query_embedding is not None and any(w in user_message.lower() for w in ["document", "doc", "file", "pdf", "csv", "txt"]):
                doc_chunks = await self.db.search_document_chunks(query_embedding, limit=5, distance_threshold=0.6)
                if doc_chunks:
                    document_context = "\n\n=== RELEVANT LOCAL DOCUMENTS ===\n"
                    for dc in doc_chunks:
                        document_context += f"Source: {dc['filename']}\nContent: {dc['text']}\n---\n"
            
            system_prompt = self._build_buddy_prompt(user_facts, model_id)
            if document_context:
                system_prompt += document_context
                
            agent = self.buddy_agent
            messages = [{"role": "system", "content": system_prompt}] + conversation_history + [{"role": "user", "content": user_message}]

        # 3. Execution
        if is_finance_workflow:
            wf_res = await finance_wf.run(user_message, model_id=model_id)
            result_content = wf_res["final_report"]
            result_model_used = model_id
            result_token_count = 0  # Trading desk manages its own internal token metrics
            fallback_occurred = False
            fallback_from = None
        else:
            result_content, result_model_used, result_token_count, fallback_occurred, fallback_from = await agent.run(messages, model_id=model_id)

        # 4. Save
        try:
            user_tokens = token_counter(model=model_id, text=user_message)
        except Exception:
            user_tokens = int(len((user_message or "").split()) * 1.3)
            
        user_msg_id = await self.db.save_message(
            conversation_id=conversation_id, role="user", content=user_message, model_id=model_id, token_count=user_tokens, keywords=""
        )
        asst_msg_id = await self.db.save_message(
            conversation_id=conversation_id, role="assistant", content=result_content, model_id=result_model_used, token_count=result_token_count, keywords=""
        )

        await self._extract_facts_regex(user_message, conversation_id, model_id=model_id)
        
        # Launch LLM fact extraction as a background task to avoid blocking the reply
        self._launch_background_task(self._extract_facts_background(user_message, model_id))

        # Generate conversation title after the 2nd full exchange (conversation_history
        # holds messages BEFORE this turn, so len == 2 means we just completed turn 2)
        if len(conversation_history) == 2:
            self._launch_background_task(
                self._generate_conversation_title(conversation_id, model_id)
            )

        await self._check_context_window(conversation_id, model_id)

        return OrchestratorResponse(
            response=result_content,
            conversation_id=conversation_id,
            extracted_facts=[],
            model_used=result_model_used,
            fallback_occurred=fallback_occurred,
            fallback_from=fallback_from,
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
            "model_id": conversation.model_id,
            "message_count": len(messages),
            "total_tokens": token_count,
        }

def create_orchestrator(router: BuddyRouter, database: BuddyDatabase) -> BuddyOrchestrator:
    return BuddyOrchestrator(router, database)
