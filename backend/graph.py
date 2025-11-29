from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from models import AgentState
from agents import (
    guardrails_node, planner_node, sql_agent_node, 
    vector_agent_node, clarification_node, summarizer_node
)

workflow = StateGraph(AgentState)

workflow.add_node("guardrails", guardrails_node)
workflow.add_node("planner", planner_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("vector_agent", vector_agent_node)
workflow.add_node("clarify", clarification_node)
workflow.add_node("summarize", summarizer_node)

workflow.add_edge(START, "guardrails")

def route_guardrails(state):
    if state.get("error"): return END
    return "planner"

workflow.add_conditional_edges("guardrails", route_guardrails)

def route_planner(state):
    step = state.get("next_step")
    if step == "sql_agent": return "sql_agent"
    if step == "vector_agent": return "vector_agent"
    if step == "clarify": return "clarify"
    if step == "general": return "summarize" # Bypass tools!
    return "summarize"

workflow.add_conditional_edges("planner", route_planner)

workflow.add_edge("sql_agent", "summarize")
workflow.add_edge("vector_agent", "summarize")
workflow.add_edge("clarify", END)
workflow.add_edge("summarize", END)

checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)