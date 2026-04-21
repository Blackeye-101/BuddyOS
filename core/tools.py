import json
import logging
import urllib.request
import urllib.parse
from xml.etree import ElementTree
import re
from typing import Dict, Any, List, Optional
from ddgs import DDGS

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, db=None, router=None, supervisor=None):
        self.db = db
        self.router = router
        self.supervisor = supervisor  # Inject supervisor for cross-agent calls
        
        self._tools = {
            "web_search": self.web_search,
            "arxiv_search": self.arxiv_search,
            "delegate_to_buddy_for_web_search": self.delegate_to_buddy_for_web_search,
        }

    def web_search(self, query: str, max_results: int = 4) -> str:
        """Search the web using DuckDuckGo."""
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

    def arxiv_search(self, query: str, max_results: int = 4) -> str:
        """Search ArXiv for academic papers with a timeout and limit."""
        logger.info(f"Executing arxiv_search for query: {query}")
        # Clean query for URL
        safe_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results={max_results}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'BuddyOS-Researcher/1.0 (Contact: user@domain.com)'})
            with urllib.request.urlopen(req, timeout=15.0) as response:
                data = response.read()
            
            root = ElementTree.fromstring(data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('.//atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else "No Title"
                
                summary_elem = entry.find('atom:summary', ns)
                summary = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None else "No Summary"
                
                link_elem = entry.find('atom:id', ns)
                link = link_elem.text if link_elem is not None else ""
                
                published_elem = entry.find('atom:published', ns)
                published = published_elem.text[:4] if published_elem is not None else "Unknown" # Just year
                
                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None]
                first_author = authors[0] if authors else "Unknown"
                
                # Truncate summary to ~150 words
                words = summary.split()
                if len(words) > 150:
                    summary = " ".join(words[:150]) + "..."
                    
                papers.append(f"Title: {title}\nAuthor: {first_author}\nYear: {published}\nLink: {link}\nAbstract: {summary}")
            
            if not papers:
                return "No papers found on ArXiv."
            return "\n\n---\n\n".join(papers)
        
        except urllib.error.URLError as e:
            logger.error(f"ArXiv API Timeout/Error: {e}")
            return f"ArXiv Search failed or timed out: {str(e)}. Try again later."
        except Exception as e:
            logger.error(f"ArXiv parsing error: {e}")
            return f"ArXiv Search failed parsing: {str(e)}."

    def delegate_to_buddy_for_web_search(self, query: str) -> str:
        """Delegates general web searches to the Buddy Agent if the Researcher needs context."""
        logger.info(f"Delegating web search to Buddy for: {query}")
        if self.supervisor:
            return self.supervisor.run_agent_sync("buddy", f"Search the web and briefly summarize the answer for this topic: {query}")
        return self.web_search(query, max_results=3) # Fallback if supervisor not injected

    def get_tools_for_agent(self, agent_name: str) -> Optional[List[Dict[str, Any]]]:
        if agent_name.lower() == "buddy":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for real-time information, recent news, or answers to questions you are unsure about.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query."}
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        elif agent_name.lower() == "researcher":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "arxiv_search",
                        "description": "Search ArXiv for peer-reviewed academic papers. Use this for rigorous research.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The academic search query."}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delegate_to_buddy_for_web_search",
                        "description": "If you need general (non-academic) information like current events or stock prices, delegate the search to the Buddy agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The general web query."}
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        return None

    def execute_tool(self, function_name: str, arguments_json: str) -> str:
        if function_name not in self._tools:
            return f"Error: Tool '{function_name}' not found."
        
        try:
            args = json.loads(arguments_json)
            result = self._tools[function_name](**args)
            return str(result)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {arguments_json}"
        except Exception as e:
            logger.error(f"Tool execution error ({function_name}): {e}")
            return f"Error executing tool: {str(e)}"
