import logging
from core.tools import plugin

logger = logging.getLogger(__name__)

@plugin(
    name="delegate_to_buddy_for_web_search",
    description="If you need general (non-academic) information like current events or stock prices, delegate the search to the Buddy agent.",
    allowed_agents=["researcher"],
    parameters={
        "query": {"type": "string", "description": "The general web query."}
    }
)
def delegate_to_buddy_for_web_search(query: str, context: dict = None) -> str:
    logger.info(f"Delegating web search to Buddy for: {query}")
    if context and context.get("supervisor"):
        # Assuming run_agent_sync is implemented inside supervisor
        return context["supervisor"].run_agent_sync("buddy", f"Search the web and briefly summarize the answer for this topic: {query}")
    return "Error: Supervisor context not available for delegation."