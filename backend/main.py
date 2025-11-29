from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from graph import app_graph
from models import ChatRequest, ChatResponse
import redis
import os

app = FastAPI(title="Invoice Agent API")
r = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Reset turn usage by passing an empty dict update to the state first?
        # LangGraph state is persistent. We can't easily "reset" part of it via invoke params directly without a state update.
        # However, our track_usage logic simply overwrites 'token_usage_turn' based on the accumulator.
        # Actually, to be accurate, we should ideally zero it out.
        # For MVP, we will rely on the fact that 'track_usage' adds to the OLD value in the state.
        # We need to manually clear 'token_usage_turn' before invoking.
        
        # NOTE: With MemorySaver, state persists. 
        # We will update the state to clear turn usage before running.
        current_state = app_graph.get_state(config)
        if current_state.values:
            app_graph.update_state(config, {"token_usage_turn": {"total":0, "prompt":0, "completion":0}})

        result = app_graph.invoke(
            {"messages": [HumanMessage(content=req.message)]}, 
            config=config
        )
        
        last_message = result['messages'][-1].content
        steps = result.get("steps_log", [])
        
        # Get the usages
        session_usage = result.get("token_usage_session", {})
        turn_usage = result.get("token_usage_turn", {})
        
        # Combine for frontend display
        usage_display = {
            "turn": turn_usage,
            "session": session_usage
        }
        
        if result.get("error"):
            return ChatResponse(response=f"Error: {result['error']}", steps=steps, token_usage=usage_display)
            
        return ChatResponse(response=last_message, steps=steps, token_usage=usage_display)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))