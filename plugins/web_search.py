import logging
from ddgs import DDGS
from core.tools import plugin

logger = logging.getLogger(__name__)

@plugin(
    name="web_search",
    description="Search the web for real-time information, recent news, or answers to questions you are unsure about.",
    allowed_agents=["buddy"],
    parameters={
        "query": {"type": "string", "description": "The search query."}
    }
)
def web_search(query: str, max_results: int = 4, context: dict = None) -> str:
    logger.info(f"Executing web_search for query: {query}")
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    
    formatted_results = []
    for r in results:
        formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}")
        
    return "\n\n".join(formatted_results)
