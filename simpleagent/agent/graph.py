import logging

from langgraph.graph import StateGraph, START, END

from ..service.state import AgentState
from ..node.router import router_node
from ..node.summarize import summarize_node
from ..node.act import act_node
from ..node.conversation import conversation_node

logger = logging.getLogger(__name__)

def create_agent_graph(model_name: str, base_url: str):
    """Creates and compiles the LangGraph StateGraph for the ReAct agent.

    Graph flow:
        START → router → [conversation | summarize | act] → END
    """
    workflow = StateGraph(AgentState)

    # Inject model config into each node using closures
    async def _router(state): return await router_node(state, model_name, base_url)
    async def _summarize(state): return await summarize_node(state, model_name, base_url)
    async def _act(state): return await act_node(state, model_name, base_url)
    async def _conversation(state): return await conversation_node(state, model_name, base_url)

    # Register nodes
    workflow.add_node("router", _router)
    workflow.add_node("summarize", _summarize)
    workflow.add_node("act", _act)
    workflow.add_node("conversation", _conversation)

    # Entry point
    workflow.add_edge(START, "router")

    # Router decision
    def route_decision(state: AgentState) -> str:
        decision = state.get("next_step", "act")
        logger.info(f"[Graph] Routing to: {decision}")
        return decision

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"summarize": "summarize", "act": "act", "conversation": "conversation"}
    )

    # All terminal nodes go straight to END
    workflow.add_edge("summarize", END)
    workflow.add_edge("act", END)
    workflow.add_edge("conversation", END)

    return workflow.compile()
