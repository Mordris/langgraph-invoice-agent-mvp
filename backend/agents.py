import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from models import AgentState
from utils import get_token_friendly_schema
from tools import run_sql_query, semantic_search

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
DB_URL = os.getenv("DATABASE_URL")

def log_step(state: AgentState, step_message: str):
    """Helper to append logs to state."""
    current_logs = state.get("steps_log", [])
    return current_logs + [step_message]

# --- Node Functions ---

def guardrails_node(state: AgentState):
    """
    Step 1: Strict relevance check.
    """
    last_msg = state['messages'][-1].content
    
    system_prompt = """
    You are a classification firewall for an Invoice Analytics Bot.
    Check if the user query is relevant to:
    1. Invoices, taxes, spending, accounting, merchants.
    2. Buying items, product details, shopping history.
    3. General conversational greetings (hello, hi, help).
    
    Reply ONLY with 'YES' or 'NO'. Do not explain. Do not add punctuation.
    """
    
    # We send the last message for classification
    response = llm.invoke([SystemMessage(content=system_prompt), state['messages'][-1]])
    decision = response.content.strip().upper()
    
    # Strict check: Only block if it explicitly says "NO"
    if decision == "NO":
        return {
            "error": "Irrelevant query.", 
            "next_step": "end_conversation",
            "steps_log": ["🛡️ Guardrails: Blocked irrelevant query."]
        }
    
    return {
        "remaining_loops": 3, 
        "error": None,
        "steps_log": ["🛡️ Guardrails: Query passed."]
    }

def planner_node(state: AgentState):
    """
    Step 2: Context-Aware Planning & Query Refinement.
    Rewrites 'that invoice' into specific IDs based on chat history.
    """
    # Get last 5 messages to provide context
    recent_history = state['messages'][-5:]
    schema = get_token_friendly_schema(DB_URL)
    
    prompt = f"""
    You are a Planner Agent.
    Schema:
    {schema}
    
    Chat History:
    {recent_history}
    
    Task:
    1. REWRITE the last user message to be a standalone, specific query.
       - Resolve "that invoice", "it", "the first one" to actual IDs or names found in the history.
       - If the query is already specific, keep it as is.
    2. CHOOSE the next tool: 'sql_agent', 'vector_agent', or 'clarify'.
    
    Return JSON: 
    {{
        "refined_query": "The actual query to run", 
        "reasoning": "Why I chose this tool", 
        "next": "sql_agent" | "vector_agent" | "clarify"
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content.replace("```json", "").replace("```", "").strip()
        decision = json.loads(content)
        
        refined = decision.get("refined_query", state['messages'][-1].content)
        
        log_msg = f"🧠 Planner: Rewrote input as: '{refined}'\n   Routing to: {decision['next']}"
        
        return {
            "plan": decision["reasoning"], 
            "refined_query": refined,
            "next_step": decision["next"],
            "steps_log": log_step(state, log_msg)
        }
    except Exception as e:
        # Fallback to vector search if parsing fails
        return {
            "refined_query": state['messages'][-1].content,
            "next_step": "vector_agent",
            "steps_log": log_step(state, f"🧠 Planner Error: {e}. Defaulting to Vector Search.")
        }

def sql_agent_node(state: AgentState):
    """
    Step 3a: Analytics via SQL (Using Refined Query).
    """
    if state.get('remaining_loops', 0) <= 0:
        return {
            "error": "Max execution loops reached.",
            "steps_log": log_step(state, "🛑 SQL Agent: Max loops reached.")
        }
        
    schema = get_token_friendly_schema(DB_URL)
    # Use the refined query from the Planner
    question = state.get('refined_query', state['messages'][-1].content)
    
    prompt = f"""
    You are a PostgreSQL Expert.
    Schema: {schema}
    
    Question: {question}
    
    Return ONLY a valid PostgreSQL SELECT query. Do not wrap in markdown.
    """
    response = llm.invoke(prompt)
    query = response.content.replace("```sql", "").replace("```", "").strip()
    
    result = run_sql_query(query)
    
    return {
        "sql_query": query,
        "sql_result": result,
        "next_step": "summarize",
        "remaining_loops": state['remaining_loops'] - 1,
        "steps_log": log_step(state, f"💾 SQL Agent: Ran Query: `{query}`\n   Rows found: {str(result)[:50]}...")
    }

def vector_agent_node(state: AgentState):
    """
    Step 3b: Semantic Search (Using Refined Query).
    """
    # Use the refined query
    question = state.get('refined_query', state['messages'][-1].content)
    result = semantic_search(question)
    
    return {
        "sql_result": result, 
        "next_step": "summarize",
        "steps_log": log_step(state, f"🔍 Vector Agent: Searched for '{question}'")
    }

def clarification_node(state: AgentState):
    return {
        "clarification_needed": True,
        "messages": [AIMessage(content="I need a bit more detail. Which specific date, merchant, or invoice are you asking about?")],
        "steps_log": log_step(state, "❓ Clarification: Asking user for details.")
    }

def summarizer_node(state: AgentState):
    context = state.get("sql_result")
    # Use refined query for context
    query = state.get('refined_query', state['messages'][-1].content)
    
    prompt = f"""
    User Query: {query}
    Data Retrieved: {context}
    
    Answer the user politely and concisely.
    """
    
    response = llm.invoke(prompt)
    return {
        "messages": [response],
        "next_step": "end",
        "steps_log": log_step(state, "📝 Summarizer: Generated response.")
    }