from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from models import AgentState
from agents import (
    guardrails_node, intent_node, refiner_node, 
    sql_agent_node, vector_agent_node, 
    clarification_node, summarizer_node
)

workflow = StateGraph(AgentState)

workflow.add_node("guardrails", guardrails_node)
workflow.add_node("intent_classifier", intent_node)
workflow.add_node("query_refiner", refiner_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("vector_agent", vector_agent_node)
workflow.add_node("clarify", clarification_node)
workflow.add_node("summarize", summarizer_node)

# Flow
workflow.add_edge(START, "guardrails")

def route_guardrails(state):
    if state.get("error"): return END
    return "intent_classifier"

workflow.add_conditional_edges("guardrails", route_guardrails)

def route_intent(state):
    intent = state.get("intent")
    if intent == "general": return "summarize"
    if intent == "clarify": return "clarify"
    return "query_refiner" # Both analytics and search go to refiner

workflow.add_conditional_edges("intent_classifier", route_intent)

def route_refiner(state):
    return state.get("next_step") # sql_agent or vector_agent

workflow.add_conditional_edges("query_refiner", route_refiner)

workflow.add_edge("sql_agent", "summarize")
workflow.add_edge("vector_agent", "summarize")
workflow.add_edge("clarify", END)
workflow.add_edge("summarize", END)

checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)