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

# --- API Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str