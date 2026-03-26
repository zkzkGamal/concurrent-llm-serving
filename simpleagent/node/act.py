import logging

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..service.state import AgentState
from ..service.llm import get_llm
from ..tools.tools import TOOLS

logger = logging.getLogger(__name__)

ACT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Always use the appropriate tool to answer the question accurately. "
    "Do not guess — if the answer requires a tool, use it."
)

async def act_node(state: AgentState, model_name: str, base_url: str) -> dict:
    """Executes a ReAct agent loop to utilize tools for answering the prompt."""
    llm = get_llm(model_name, base_url)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ACT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False, handle_parsing_errors=True)

    history = list(state["messages"][:-1])
    last_msg = state["messages"][-1].content

    try:
        res = await executor.ainvoke({"input": last_msg, "chat_history": history})
        output = res["output"]
    except Exception as e:
        logger.error(f"[Act] Tool execution failed: {e}")
        output = f"I encountered an error while using tools: {str(e)}"

    return {"messages": [AIMessage(content=output)]}
