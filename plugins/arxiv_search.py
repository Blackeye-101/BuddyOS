import logging
import urllib.request
import urllib.parse
from xml.etree import ElementTree
from core.tools import plugin

logger = logging.getLogger(__name__)

@plugin(
    name="arxiv_search",
    description="Search ArXiv for peer-reviewed academic papers. Use this for rigorous research.",
    allowed_agents=["researcher"],
    parameters={
        "query": {"type": "string", "description": "The academic search query."}
    }
)
def arxiv_search(query: str, max_results: int = 4, context: dict = None) -> str:
    logger.info(f"Executing arxiv_search for query: {query}")
    safe_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results={max_results}"
    
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
        
        words = summary.split()
        if len(words) > 150:
            summary = " ".join(words[:150]) + "..."
            
        papers.append(f"Title: {title}\nAuthor: {first_author}\nYear: {published}\nLink: {link}\nAbstract: {summary}")
    
    if not papers:
        return "No papers found on ArXiv."
    return "\n\n---\n\n".join(papers)
