from typing import TypedDict, List, Optional, Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

# --- LangGraph State ---
class AgentState(TypedDict):
    """
    The shared state accessible by all agents in the graph.
    'messages' is append-only (managed by add_messages).
    Other fields are overwritten by the agents.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Seperation of intent vs plan
    intent: Optional[str]        # 'general', 'sql', 'vector', 'clarify'
    refined_query: Optional[str] # The rewritten standalone query

    # Routing logic
    next_step: Optional[str]
    remaining_loops: int
    
    # Planner outputs
    plan: Optional[str]
    
    # SQL Agent outputs
    sql_query: Optional[str]
    sql_result: Optional[str]
    
    # Flags & Errors
    clarification_needed: Optional[bool]
    error: Optional[str]
    
    # Collect logs/steps to show the user
    steps_log: List[str] 

    # Track accumulated token usage
    token_usage_session: dict # Cumulative
    token_usage_turn: dict    # Just this interaction

# --- API Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    steps: List[str] = []
    token_usage: dict = {}