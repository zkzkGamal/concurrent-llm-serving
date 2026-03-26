import logging
from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from ..service.state import AgentState
from ..service.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a routing assistant. Analyze the user's last message carefully.
Route to exactly one of the following:
- 'summarize': if the user explicitly wants text, a document, or context summarized.
- 'act': if the user needs a tool — searching the web, math calculations, checking the time, or fetching a document.
- 'conversation': if the user is chatting casually or asking a simple conversational question.
Reply ONLY with the JSON structure requested, nothing else."""

class Route(BaseModel):
    next_node: Literal["summarize", "act", "conversation"] = Field(
        description="The next node to route to based on the user's request."
    )

async def router_node(state: AgentState, model_name: str, base_url: str) -> dict:
    """Analyzes the conversation and routes to the appropriate next node."""
    llm = get_llm(model_name, base_url)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    try:
        structured_llm = llm.with_structured_output(Route)
        result = await structured_llm.ainvoke(messages)
        next_node = result.next_node
    except Exception as e:
        logger.warning(f"[Router] Structured output failed ({e}), defaulting to 'act'")
        next_node = "act"

    logger.info(f"[Router] → '{next_node}'")
    return {"next_step": next_node}
