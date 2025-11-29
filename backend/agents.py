import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from models import AgentState
from utils import get_token_friendly_schema
from tools import run_sql_query, semantic_search

# LLM Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
DB_URL = os.getenv("DATABASE_URL")

# --- Helper Functions ---

def log_step(state: AgentState, step_message: str):
    """
    Helper to append logs to state so they can be sent to frontend.
    """
    current_logs = state.get("steps_log", [])
    return current_logs + [step_message]

# --- Node Functions ---

def guardrails_node(state: AgentState):
    """
    Step 1: Verify if the input is safe and relevant.
    """
    last_msg = state['messages'][-1].content
    
    system_prompt = """
    You are a firewall. Your job is to check if the user query is relevant to:
    1. Invoices, payments, taxes, merchants.
    2. Buying items, shopping history.
    3. General greetings (hello, hi).
    
    If RELEVANT, reply 'YES'.
    If IRRELEVANT (e.g. asking about celebrities, coding help unrelated to this app), reply 'NO'.
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt), state['messages'][-1]])
    decision = response.content.strip().upper()
    
    if "NO" in decision:
        return {
            "error": "Irrelevant query.", 
            "next_step": "end_conversation",
            "steps_log": ["🛡️ Guardrails: Query deemed irrelevant."]
        }
    
    # Initialize loop counter for the new turn
    return {
        "remaining_loops": 3, 
        "error": None,
        "steps_log": ["🛡️ Guardrails: Query passed relevance check."]
    }

def planner_node(state: AgentState):
    """
    Step 2: Analyze the user's intent and choose the tool.
    """
    query = state['messages'][-1].content
    schema = get_token_friendly_schema(DB_URL)
    
    prompt = f"""
    You are a Planner. You have access to an Invoice Database.
    Schema:
    {schema}
    
    User Query: "{query}"
    
    Task: specificy the next step.
    1. 'sql_agent': For aggregations (total, average, count), specific dates, or math.
    2. 'vector_agent': For finding specific items ("Samsung phone"), merchant details, or vague text searches.
    3. 'clarify': If the user query is too vague (e.g. "How much?" without saying what).
    
    Return JSON format: {{"reasoning": "...", "next": "sql_agent" | "vector_agent" | "clarify"}}
    """
    
    try:
        response = llm.invoke(prompt)
        # Basic JSON parsing (in prod use structured output)
        content = response.content.replace("```json", "").replace("```", "").strip()
        decision = json.loads(content)
        
        step_log_msg = f"🧠 Planner: {decision['reasoning']} -> Routing to {decision['next']}"
        
        return {
            "plan": decision["reasoning"], 
            "next_step": decision["next"],
            "steps_log": log_step(state, step_log_msg)
        }
    except Exception:
        # Fallback
        return {
            "plan": "Defaulting to vector search due to parse error", 
            "next_step": "vector_agent",
            "steps_log": log_step(state, "🧠 Planner: JSON parse error, defaulting to Vector Agent.")
        }

def sql_agent_node(state: AgentState):
    """
    Step 3a: Analytics via SQL.
    """
    if state.get('remaining_loops', 0) <= 0:
        return {
            "error": "Max execution loops reached.",
            "steps_log": log_step(state, "🛑 SQL Agent: Max loops reached.")
        }
        
    schema = get_token_friendly_schema(DB_URL)
    question = state['messages'][-1].content
    
    # Generate SQL
    prompt = f"""
    You are a PostgreSQL Expert.
    Schema: {schema}
    
    Question: {question}
    
    Return ONLY a valid PostgreSQL SELECT query. Do not wrap in markdown.
    """
    response = llm.invoke(prompt)
    query = response.content.replace("```sql", "").replace("```", "").strip()
    
    # Execute
    result = run_sql_query(query)
    
    log_msg = f"💾 SQL Agent: Generated SQL: `{query}`\n   Result: {str(result)[:100]}..."
    
    return {
        "sql_query": query,
        "sql_result": result,
        "next_step": "summarize",
        "remaining_loops": state['remaining_loops'] - 1,
        "steps_log": log_step(state, log_msg)
    }

def vector_agent_node(state: AgentState):
    """
    Step 3b: Semantic Search.
    """
    question = state['messages'][-1].content
    result = semantic_search(question)
    
    log_msg = f"🔍 Vector Agent: Performed semantic search.\n   Found: {result[:100]}..."
    
    return {
        "sql_result": result, # We use the same field 'sql_result' to store data for the summarizer
        "next_step": "summarize",
        "steps_log": log_step(state, log_msg)
    }

def clarification_node(state: AgentState):
    """
    Step 3c: Ask User for info.
    """
    return {
        "clarification_needed": True,
        # The graph will interrupt here, the frontend will display this message
        "messages": [AIMessage(content="I need more details. Could you specify the date range, merchant, or item name?")],
        "steps_log": log_step(state, "❓ Clarification: Asking user for more context.")
    }

def summarizer_node(state: AgentState):
    """
    Step 4: Formulate the answer.
    """
    context = state.get("sql_result")
    query = state['messages'][-1].content
    
    prompt = f"""
    User Question: {query}
    Data Retrieved: {context}
    
    Answer the user politely and concisely based on the data. 
    If the data says 'No results', tell the user you couldn't find that info.
    """
    
    response = llm.invoke(prompt)
    return {
        "messages": [response],
        "next_step": "end",
        "steps_log": log_step(state, "📝 Summarizer: Formatting final answer.")
    }