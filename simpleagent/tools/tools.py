import logging
import math
from datetime import datetime

from langchain_core.tools import tool
import ddgs

logger = logging.getLogger(__name__)

@tool
def get_current_time() -> str:
    """Returns the current date and time in ISO format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_expression(expression: str) -> str:
    """Calculates a safe mathematical expression.
    Example: '2 + 2' or 'math.sqrt(16)'
    Supports standard math functions like sqrt, pow, log, etc.
    """
    try:
        safe_globals = {"__builtins__": None}
        safe_locals = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        result = eval(expression, safe_globals, safe_locals)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results.
    Use this for any question requiring up-to-date information.
    """
    try:
        results = []
        with ddgs.DDGS() as search:
            for r in search.text(query, max_results=max_results):
                logger.info(f"[DDGS] Result: {r.get('title','?')}")
                results.append(
                    f"**{r.get('title', 'No Title')}**\n{r.get('body', '')}\nURL: {r.get('href', '')}"
                )
        if not results:
            return "No results found for this query."
        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.error(f"[DDGS] Search error: {e}")
        return f"Error searching the web: {e}"

@tool
def get_document_context(doc_name: str) -> str:
    """Returns simulated long-form content for a named document.
    Used for testing long-context processing.
    """
    paragraphs = [
        f"## {doc_name} - Section {i+1}\n"
        f"This section covers key aspects of the topic including performance characteristics, "
        f"scalability considerations, and best practices for deployment in production environments. "
        f"Engineers should pay attention to memory utilization and GPU scheduling when working with "
        f"large batch sizes in distributed inference systems."
        for i in range(10)
    ]
    return "\n\n".join(paragraphs)

TOOLS = [get_current_time, calculate_expression, web_search, get_document_context]
