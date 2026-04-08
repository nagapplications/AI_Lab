# Multi-Agent Debate System:
#
# They interact like:
#
# User Question
#    ↓
# Agent A → gives answer
#    ↓
# Agent B → critiques / finds flaws
#    ↓
# Agent A → improves answer
#    ↓
# Judge → picks best or final answer

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from setup.config import OPENAI_API_KEY

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../setup/.env"))

# --------------------- STEP 1 : Define state object  --------------------------------------
class DebateState(TypedDict):
    question: str
    proposer_answer: str
    critic_feedback: str
    judge_decision: str

# --------------------- STEP 2 : Define Agent Prompts--------------------------------------
proposer_prompt = SystemMessage(content="""
    You are a helpful assistant, Provide a clear and complete answer
""")
critic_prompt = SystemMessage(content="""
    You are a critical assistant, Your job is to find flaws, gaps, or weaknesses in the answer provided by the proposer.
""")
judge_prompt = SystemMessage(content="""
    You are a fair judge, Your job is to evaluate the answer and feedback, and make a final decision on the best answer.
""")

# --------------------- STEP 3 : Define Nodes --------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

def proposer_node(state: DebateState) -> DebateState:
    print("\n🧭 Proposer Node proposing...")
    response = llm.invoke([proposer_prompt, HumanMessage(content=state["question"])])
    print(response.content)
    return {"proposer_answer": response.content}

def critic_node(state: DebateState) -> DebateState:
    print("\n🧭 Critic Node gives feedback...")
    response = llm.invoke([critic_prompt, HumanMessage(content=state["proposer_answer"])])
    print(response.content)
    return {"critic_feedback": response.content}

def judge_node(state: DebateState) -> DebateState:
    print("\n🧭 Judge Node evaluating...")
    response = llm.invoke([judge_prompt,
                           HumanMessage(content=f"""
                               Question: {state['question']}
                               Answer: {state['proposer_answer']}
                               Feedback: {state['critic_feedback']}
                                
                               Is this answer good enough? Reply YES or NO.
                           """)
                           ])
    print(response.content)
    return {"judge_decision": response.content}

def refine_node(state):
    print("\n🔁 Proposer refining...")
    response = llm.invoke([
        proposer_prompt,
        HumanMessage(content=f"""
            Question: {state['question']}
            Previous Answer: {state['proposer_answer']}
            Critic Feedback: {state['critic_feedback']}
            Improve the answer.
        """)
    ])
    print(response.content)
    return {"proposer_answer": response.content}

def should_refine(state):
    if "NO" in state["judge_decision"]:
        return "refine"
    return "end"

# --------------------- STEP 4 : Build Graph & Compile --------------------------------------
graph_builder = StateGraph(DebateState)
graph_builder.add_node("propose", proposer_node)
graph_builder.add_node("feedback", critic_node)
graph_builder.add_node("judge", judge_node)
graph_builder.add_node("refine", refine_node)

graph_builder.add_edge(START, "propose")
graph_builder.add_edge("propose", "feedback")
graph_builder.add_edge("feedback", "judge")
graph_builder.add_conditional_edges("judge", should_refine, {"refine": "refine", "end": END})

graph = graph_builder.compile()

# --------------------- STEP 5 : RUN --------------------------------------
def run(user_input: str):
    print(f"\n{'-' * 50}")
    print(f"User: {user_input}")
    print(f"{'-' * 50}")
    result = graph.invoke({"question": user_input})
    final_answer = result["proposer_answer"]
    print(f"\n✅ Final Answer: {final_answer}")

# run("What is a good recipe for chocolate chip cookies?")
# run("How to impress the girl in the first date?")
run("How to overcome over thinking?")
