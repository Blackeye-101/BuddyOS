import json
import logging
from typing import Dict, Any, List, Optional
from ddgs import DDGS

logger = logging.getLogger(__name__)

def web_search(query: str, max_results: int = 4) -> str:
    """
    Search the web using DuckDuckGo.
    """
    logger.info(f"Executing web_search for query: {query}")
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in web_search: {e}")
        return f"Search failed: {str(e)}"

# Define available tools and their schemas
BUDDY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information, recent news, or answers to questions you are unsure about. Do not use this for general conversational responses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g. 'latest AI news', 'weather in Tokyo')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Create a mapping for execution
_TOOL_FUNCTIONS = {
    "web_search": web_search
}

def get_tools_for_agent(agent_name: str) -> Optional[List[Dict[str, Any]]]:
    """Return the tool schema array for the given agent."""
    if agent_name.lower() == "buddy":
        return BUDDY_TOOLS
    # Future extensibility for "researcher" etc
    return None

def execute_tool(function_name: str, arguments_json: str) -> str:
    """Execute a tool dynamically based on name and json string arguments."""
    if function_name not in _TOOL_FUNCTIONS:
        return f"Error: Tool '{function_name}' not found."
    
    try:
        args = json.loads(arguments_json)
        # Call the corresponding Python function
        result = _TOOL_FUNCTIONS[function_name](**args)
        return str(result)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON arguments: {arguments_json}"
    except Exception as e:
        logger.error(f"Tool execution error ({function_name}): {e}")
        return f"Error executing tool: {str(e)}"
