from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from models import AgentState
from agents import (
    guardrails_node, 
    planner_node, 
    sql_agent_node, 
    vector_agent_node, 
    clarification_node, 
    summarizer_node
)

# 1. Initialize Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("guardrails", guardrails_node)
workflow.add_node("planner", planner_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("vector_agent", vector_agent_node)
workflow.add_node("clarify", clarification_node)
workflow.add_node("summarize", summarizer_node)

# 3. Define Edges & Routing Logic

# Start -> Guardrails
workflow.add_edge(START, "guardrails")

# Guardrails -> Planner (or End if irrelevant)
def route_guardrails(state):
    if state.get("error") == "Irrelevant query.":
        return END
    return "planner"

workflow.add_conditional_edges("guardrails", route_guardrails)

# Planner -> specific Agent
def route_planner(state):
    step = state.get("next_step")
    if step == "sql_agent": return "sql_agent"
    if step == "vector_agent": return "vector_agent"
    return "clarify"

workflow.add_conditional_edges("planner", route_planner)

# Agents -> Summarize
workflow.add_edge("sql_agent", "summarize")
workflow.add_edge("vector_agent", "summarize")

# Summarize -> End
workflow.add_edge("summarize", END)

# Clarify -> End (Interruption)
# When we hit 'clarify', the graph stops and returns the question to the user.
# The user's next message effectively resumes the session.
workflow.add_edge("clarify", END)

# 4. Compile Graph
# MemorySaver is used here for MVP. In a cluster, use RedisSaver (not yet standard in LangGraph MVP)
# or implement a custom Checkpointer. For this MVP, MemorySaver persists state in RAM per session.
checkpointer = MemorySaver()

# We interrupt before 'clarify' effectively (or rather, the node executes, sends msg, then ends).
# If we wanted the user to answer *into* the clarify node, we'd interrupt before.
# Here, simpler pattern: Clarify node outputs a message and graph ends. 
# Next user message starts a new run with history.
app_graph = workflow.compile(checkpointer=checkpointer)