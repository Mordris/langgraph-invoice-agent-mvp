from typing import TypedDict, List, Optional, Annotated, Any, Dict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

# --- LangGraph State ---
class AgentState(TypedDict):
    """
    The shared state accessible by all agents in the graph.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Intent & Planning
    intent: Optional[str]
    refined_query: Optional[str]

    # Routing
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
    
    # Logs
    steps_log: List[str] 

    # Token tracking
    token_usage_session: dict
    token_usage_turn: dict
    
    # NEW: Performance tracking
    performance: Dict[str, Dict[str, float]]  # {agent_name: {metric: value}}

# --- API Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class PerformanceMetrics(BaseModel):
    agent_name: str
    total_time: float
    llm_time: Optional[float] = None
    db_time: Optional[float] = None
    search_time: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    steps: List[str] = []
    token_usage: dict = {}
    performance: List[PerformanceMetrics] = []  # NEW: Performance data
    total_execution_time: float = 0.0  # NEW: Overall time