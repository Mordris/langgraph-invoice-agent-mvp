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
        # FIXED: Properly reset turn usage by including it in the invoke call
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "token_usage_turn": {"total": 0, "prompt": 0, "completion": 0}
        }
        
        result = app_graph.invoke(initial_state, config=config)
        
        last_message = result['messages'][-1].content
        steps = result.get("steps_log", [])
        
        # Get the usages
        session_usage = result.get("token_usage_session", {"total": 0, "prompt": 0, "completion": 0})
        turn_usage = result.get("token_usage_turn", {"total": 0, "prompt": 0, "completion": 0})
        
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