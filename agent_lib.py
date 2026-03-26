import os
from typing import List, Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# ---------------------------
# 1️⃣ Define Tools
# ---------------------------

@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_expression(expression: str) -> str:
    """Calculates a mathematical expression.
    Example: '2 + 2' or 'math.sqrt(16)'
    """
    try:
        # Using eval safely-ish for simple math
        return str(eval(expression, {"__builtins__": None}, {"math": __import__("math")}))
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Simulates a web search. Returns results of varying lengths."""
    # Simulate a "long" result for specific queries
    if "deep" in query.lower() or "detailed" in query.lower():
        content = " ".join(["This is a very detailed research paper content about AI throughput and latency optimization in distributed systems."] * 100)
        return f"Long Search Result for '{query}': {content}"
    return f"Quick Search Result for '{query}': AI models are getting faster in 2026."

@tool
def get_document_context(doc_name: str) -> str:
    """Returns the full text of a document. Used for long-context testing."""
    # Simulate a 5000-character document
    content = f"Document: {doc_name}\n" + ("This is repetitive context to test model's ability to handle large input batches. " * 100)
    return content

# ---------------------------
# 2️⃣ Agent Class
# ---------------------------

class LLMAgent:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3.5-0.8B", 
                 base_url: str = "http://127.0.0.1:8000/v1",
                 system_prompt: str = "You are a helpful AI assistant with access to tools."):
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_base=base_url,
            openai_api_key="dummy",
            temperature=0, # Deterministic for testing
        )
        
        self.tools = [get_current_time, calculate_expression, web_search, get_document_context]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Note: Using create_openai_functions_agent assumes the local model supports function calling
        # If it doesn't, we might need create_react_agent or a simple custom loop.
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)
        self.history: List[BaseMessage] = []

    async def run(self, user_input: str) -> Dict[str, Any]:
        start_time = datetime.now()
        
        try:
            result = await self.executor.ainvoke({
                "input": user_input,
                "chat_history": self.history
            })
            
            output = result["output"]
            self.history.append(HumanMessage(content=user_input))
            self.history.append(AIMessage(content=output))
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "output": output,
                "duration": duration,
                "success": True,
                "tool_calls": len(self.executor.tools) # Simplified, executor doesn't easily expose call count in result
            }
        except Exception as e:
            end_time = datetime.now()
            return {
                "error": str(e),
                "duration": (end_time - start_time).total_seconds(),
                "success": False
            }

if __name__ == "__main__":
    import asyncio
    
    async def test():
        agent = LLMAgent()
        print("Testing Agent...")
        res = await agent.run("What time is it and what is 25 * 4?")
        print(f"Response: {res['output']}")
        print(f"Time taken: {res['duration']:.2f}s")
        
    asyncio.run(test())
