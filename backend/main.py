from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from graph import app_graph
from models import ChatRequest, ChatResponse, PerformanceMetrics
from agents import FastPath
import redis
import os
import time

app = FastAPI(title="Invoice Agent API")
r = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    config = {"configurable": {"thread_id": session_id}}
    request_start_time = time.time()
    
    # CRITICAL OPTIMIZATION: Fast path for greetings
    fast_response = FastPath.check(req.message)
    if fast_response:
        total_time = time.time() - request_start_time
        return ChatResponse(
            response=fast_response,
            steps=["⚡ Fast Path: Instant response (0 tokens, 0 LLM calls)"],
            token_usage={
                "turn": {"total": 0, "prompt": 0, "completion": 0},
                "session": {"total": 0, "prompt": 0, "completion": 0}
            },
            performance=[PerformanceMetrics(
                agent_name="fast_path",
                total_time=total_time,
                llm_time=0.0
            )],
            total_execution_time=total_time
        )
    
    # Normal agent pipeline
    try:
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "token_usage_turn": {"total": 0, "prompt": 0, "completion": 0},
            "performance": {}
        }
        
        result = app_graph.invoke(initial_state, config=config)
        total_execution_time = time.time() - request_start_time
        
        last_message = result['messages'][-1].content
        steps = result.get("steps_log", [])
        
        session_usage = result.get("token_usage_session", {"total": 0, "prompt": 0, "completion": 0})
        turn_usage = result.get("token_usage_turn", {"total": 0, "prompt": 0, "completion": 0})
        
        # Extract performance metrics
        performance_data = result.get("performance", {})
        performance_list = []
        
        for agent_name, metrics in performance_data.items():
            performance_list.append(PerformanceMetrics(
                agent_name=agent_name,
                total_time=metrics.get("total_time", 0),
                llm_time=metrics.get("llm_time"),
                db_time=metrics.get("db_time"),
                search_time=metrics.get("search_time")
            ))
        
        usage_display = {"turn": turn_usage, "session": session_usage}
        
        if result.get("error"):
            return ChatResponse(
                response=f"Error: {result['error']}",
                steps=steps,
                token_usage=usage_display,
                performance=performance_list,
                total_execution_time=total_execution_time
            )
        
        return ChatResponse(
            response=last_message,
            steps=steps,
            token_usage=usage_display,
            performance=performance_list,
            total_execution_time=total_execution_time
        )
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))