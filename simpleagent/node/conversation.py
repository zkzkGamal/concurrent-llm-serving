import logging

from langchain_core.messages import AIMessage, SystemMessage

from ..service.state import AgentState
from ..service.llm import get_llm

logger = logging.getLogger(__name__)

CONVERSATION_SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable AI assistant. "
    "Engage naturally with the user. Keep your replies concise and helpful."
)

async def conversation_node(state: AgentState, model_name: str, base_url: str) -> dict:
    """Handles standard, casual conversation without requiring tools."""
    llm = get_llm(model_name, base_url)
    messages = [SystemMessage(content=CONVERSATION_SYSTEM_PROMPT)] + list(state["messages"])

    try:
        res = await llm.ainvoke(messages)
        output = res.content
    except Exception as e:
        logger.error(f"[Conversation] Failed: {e}")
        output = f"Conversation failed: {str(e)}"

    return {"messages": [AIMessage(content=output)]}
