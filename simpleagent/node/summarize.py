import logging

from langchain_core.messages import AIMessage, SystemMessage

from ..service.state import AgentState
from ..service.llm import get_llm

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = (
    "You are an expert summarizer. Produce a concise, structured summary of the provided content. "
    "Use bullet points for key insights. Focus on accuracy and clarity. "
    "If the content contains metrics or data, highlight them."
)

async def summarize_node(state: AgentState, model_name: str, base_url: str) -> dict:
    """Summarizes large blocks of text or context provided in the conversation."""
    llm = get_llm(model_name, base_url)
    messages = [SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT)] + list(state["messages"])

    try:
        res = await llm.ainvoke(messages)
        output = res.content
    except Exception as e:
        logger.error(f"[Summarize] Failed: {e}")
        output = f"Summarization failed: {str(e)}"

    return {"messages": [AIMessage(content=output)]}
