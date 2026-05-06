import logging
from core.tools import plugin

logger = logging.getLogger(__name__)

@plugin(
    name="delegate_to_researcher_for_academic",
    description="If you need rigorously peer-reviewed academic papers on a topic, delegate to the Researcher.",
    allowed_agents=["buddy"],
    parameters={
        "query": {"type": "string", "description": "The academic topic to research."}
    }
)
def delegate_to_researcher_for_academic(query: str, context: dict = None) -> str:
    logger.info(f"Delegating academic research to Researcher for: {query}")
    if context and context.get("supervisor"):
        return context["supervisor"].run_agent_sync("researcher", f"Please research the latest academic papers on: {query}")
    return "Error: Supervisor context not available for delegation."