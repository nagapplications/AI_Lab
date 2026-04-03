import os
import sys
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from setup.config import OPENAI_API_KEY

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../setup/.env"))

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city} weather is 22°C and sunny"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression like '15 + 30'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [get_weather, calculate]
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

def planner_node(state: State) -> State:
    print("\n🧭 Planner Node thinking...")
    prompt = [
        HumanMessage(content="Break this task into steps:\n" + state["messages"][-1].content)
    ]
    response = llm.invoke(prompt)
    return {"messages": [response]}

def llm_node(state: State) -> State:
    print("\n🧠 LLM Node thinking...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_map = {t.name: t for t in tools}
def tools_node(state: State) -> State:
    print("\n⚡ Tools Node executing...")
    last_message = state["messages"][-1]  # get LLM's last response
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"  🔧 Calling: {tool_name} → {tool_args}")
        result = tool_map[tool_name].invoke(tool_args)
        print(f"  📦 Result: {result}")
        results.append(ToolMessage(content=str(result),tool_call_id=tool_call["id"]))

    return {"messages": results}

def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# -----------------------------------------------------------
# BUILD THE GRAPH
graph_builder = StateGraph(State)
graph_builder.add_node("planner_node", planner_node)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tools_node)

graph_builder.add_edge(START, "planner_node")
graph_builder.add_edge("planner_node", "llm")
graph_builder.add_conditional_edges("llm",should_continue,{"tools": "tools","end": END})
graph_builder.add_edge("tools", "llm")

graph = graph_builder.compile()
# -----------------------------------------------------------

# RUN
def run(user_input: str):
    print(f"\n{'=' * 50}")
    print(f"User: {user_input}")
    print(f"{'=' * 50}")
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    final_answer = result["messages"][-1].content
    print(f"\n✅ Final Answer: {final_answer}")
# -----------------------------------------------------------

run("What is the weather in Tokyo, and calculate how many degrees it is away from 50°C and also calculate result + 10?")
