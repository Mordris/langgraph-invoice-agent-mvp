from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from graph import app_graph
from models import ChatRequest, ChatResponse
import redis
import os
import uuid

app = FastAPI(title="Invoice Agent API")

# Redis for Rate Limiting
r = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
MAX_REQUESTS = int(os.getenv("MAX_QUESTIONS_PER_MIN", 5))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    
    # --- 1. Rate Limiting ---
    rate_key = f"rate_limit:{session_id}"
    current_count = r.incr(rate_key)
    if current_count == 1:
        r.expire(rate_key, 60) # Reset every minute
        
    if current_count > MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")

    # --- 2. Process with LangGraph ---
    # config defines the thread/session ID for memory
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # We invoke the graph with the new user message
        # The graph loads previous state automatically via checkpointer
        result = app_graph.invoke(
            {"messages": [HumanMessage(content=req.message)]}, 
            config=config
        )
        
        # Extract the final response (last message)
        # Note: If graph ended at 'guardrails' with error, we handle that.
        
        if result.get("error"):
            return ChatResponse(response=f"Error: {result['error']}")
            
        last_message = result['messages'][-1]
        return ChatResponse(response=last_message.content)

    except Exception as e:
        print(f"Error processing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)