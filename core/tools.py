import json
import logging
import importlib
import pkgutil
import inspect
import time
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class ToolMetadata:
    def __init__(self, name: str, description: str, allowed_agents: List[str], parameters: dict, func: Callable):
        self.name = name
        self.description = description
        self.allowed_agents = allowed_agents
        self.parameters = parameters
        self.func = func

def plugin(name: str, description: str, allowed_agents: List[str], parameters: dict):
    """Decorator to register a function as a BuddyOS Plugin/Tool."""
    def decorator(func: Callable):
        func.__tool_metadata__ = ToolMetadata(name, description, allowed_agents, parameters, func)
        return func
    return decorator

class ToolRegistry:
    def __init__(self, db=None, router=None, supervisor=None):
        self.db = db
        self.router = router
        self.supervisor = supervisor
        
        self._tools: Dict[str, ToolMetadata] = {}
        self._load_plugins()

    def _load_plugins(self):
        """Dynamically load all plugins from the plugins directory."""
        import plugins
        logger.info("Loading plugins...")
        
        for _, module_name, _ in pkgutil.iter_modules(plugins.__path__):
            full_module_name = f"plugins.{module_name}"
            module = importlib.import_module(full_module_name)
            
            for item_name, item in inspect.getmembers(module):
                if inspect.isfunction(item) and hasattr(item, '__tool_metadata__'):
                    metadata: ToolMetadata = item.__tool_metadata__
                    self._tools[metadata.name] = metadata
                    logger.info(f"Registered plugin: {metadata.name} (Agents: {metadata.allowed_agents})")

    def get_tools_for_agent(self, agent_name: str) -> Optional[List[Dict[str, Any]]]:
        """Return the JSON schema list of tools allowed for this agent."""
        agent_tools = []
        for name, meta in self._tools.items():
            if agent_name.lower() in [a.lower() for a in meta.allowed_agents]:
                agent_tools.append({
                    "type": "function",
                    "function": {
                        "name": meta.name,
                        "description": meta.description,
                        "parameters": {
                            "type": "object",
                            "properties": meta.parameters,
                            "required": list(meta.parameters.keys())
                        }
                    }
                })
        return agent_tools if agent_tools else None

    def execute_tool(self, function_name: str, arguments_json: str) -> str:
        """Execute a tool with automatic retry logic for transient failures."""
        if function_name not in self._tools:
            return f"Error: Tool '{function_name}' not found."
        
        metadata = self._tools[function_name]
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {arguments_json}"

        # Inject context dependencies so plugins can optionally use them
        if 'context' in inspect.signature(metadata.func).parameters:
            args['context'] = {
                'db': self.db,
                'router': self.router,
                'supervisor': self.supervisor
            }

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                result = metadata.func(**args)
                return str(result)
            except Exception as e:
                logger.warning(f"Tool '{function_name}' execution failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Tool '{function_name}' failed completely. Error: {e}")
                    return f"Error executing tool '{function_name}': {str(e)}"
                
                # Exponential backoff
                time.sleep(base_delay * (2 ** attempt))
                
        return f"Error: Unknown failure executing tool '{function_name}'"
