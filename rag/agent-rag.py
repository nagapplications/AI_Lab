import json
import os
import re
import sys
from typing import TypedDict, Dict, Annotated, Any, Callable

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from setup.config import OPENAI_API_KEY

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../setup/.env"))

# --------------------- DEFINE TOOLS, REGISTER THEM IN REGISTRY & BIND TO LLM--------------------------------------
TOOL_REGISTRY: Dict[str, Callable] = {}
def register_tool(tool_func):
    """Decorator to register a tool in the registry."""
    TOOL_REGISTRY[tool_func.__ne__] = tool_func
    return tool_func

@register_tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city} weather is 22°C and sunny"

@register_tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression like '15 + 30'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@register_tool
@tool
def retrieve_info(query: str) -> str:
    """Retrieve relevant information from knowledge base."""
    # ---------- Step1 : Load data ------------------------
    loader = TextLoader("data.txt")
    documents = loader.load()

    # ---------- Step2 : configure data splitter & create chunks ------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        separators=["\n", ". ", " "]
    )
    docs = text_splitter.split_documents(documents)

    # ---------- Step3 : create vectorstore ----------------------
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

tools = list(TOOL_REGISTRY.values())
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# --------------------- DEFINE STATE OBJECT --------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]  # existing (keep this)
    plan: list[Dict[str, Any]]  # structured steps from planner
    current_step: int  # which step is being executed
    last_result: str

# --------------------- DEFINE NODES --------------------------------------
def planner_node(state: State) -> State:
    print("\n🧭 Planner Node thinking...")
    tools_info = [
        {
            "name": t.name,
            "description": t.description,
            "args": list(t.args_schema.model_json_schema()["properties"].keys())
        }
        for t in tools
    ]
    print(tools_info)
    prompt = [
        HumanMessage(content=state["messages"][-1].content),
        SystemMessage(content=f"""
            You can only use the following tools. 
            Use their exact names and argument keys:
            {json.dumps(tools_info, indent=2)}
            Use retrieve_info → for factual questions from documents
            Output your plan as JSON steps using these exact names.
            Do not include explanations, markdown, or backticks. JSON ONLY.
            Each step must have a "name" (tool name) and "args" (dict of arguments).
            """)
    ]
    response = llm.invoke(prompt)
    plan_json = convertToJson(response)
    # print(plan_json)

    return {
        "messages": state["messages"],
        "plan": plan_json.get("steps", []),
        "current_step": 0,
        "last_result": ""
    }

def llm_node(state: State) -> State:
    print("\n🧠 LLM Node acting...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tools_node(state: State) -> State:
    print("\n⚡ Tools Node executing...")
    last_message = state["messages"][-1]  # The latest message determines the next action
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"  🔧 Calling: {tool_name} → {tool_args}")
        result = safe_invoke(tool_name, tool_args)
        print(f"  📦 Result: {result}")
        results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    return {"messages": results}

def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# --------------------- UTIL FUNCTIONS --------------------------------------
def convertToJson(response: AIMessage) -> Any:
    # Parse JSON safely
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.MULTILINE)
    try:
        plan_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse JSON:", e)
        plan_json = {"steps": []}

    # Normalize steps
    for step in plan_json.get("steps", []):
        if "args" not in step or not isinstance(step["args"], dict):
            step["args"] = {}
    return plan_json

tool_map = {t.name: t for t in tools}
def safe_invoke(tool_name, tool_args):
    if tool_name not in tool_map:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return tool_map[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool Error: {str(e)}"

# --------------------- BUILD GRAPH & COMPILE --------------------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("planner_node", planner_node)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tools_node)

graph_builder.add_edge(START, "planner_node")
graph_builder.add_edge("planner_node", "llm")
graph_builder.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "llm")

graph = graph_builder.compile()

# --------------------- RUN --------------------------------------
def run(user_input: str):
    print(f"\n{'=' * 50}")
    print(f"User: {user_input}")
    print(f"{'=' * 50}")
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    final_answer = result["messages"][-1].content
    print(f"\n✅ Final Answer: {final_answer}")

#run("What is the weather in Tokyo, and calculate how many degrees it is away from 50°C and also calculate result + 10?")
run("does zip knows swimming?")
