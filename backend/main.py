from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from graph import app_graph
from models import ChatRequest, ChatResponse, PerformanceMetrics
import redis
import os
import time
import json
import asyncio

app = FastAPI(title="Invoice Agent API")
r = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

async def generate_events(session_id: str, message: str):
    """
    Generator that yields Server-Sent Events for real-time updates.
    """
    config = {"configurable": {"thread_id": session_id}}
    request_start_time = time.time()
    
    try:
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': '🛡️ Checking Safety...'})}\n\n"
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "token_usage_turn": {"total": 0, "prompt": 0, "completion": 0},
            "performance": {}
        }
        
        # Stream through the graph
        last_step = None
        for state in app_graph.stream(initial_state, config=config):
            # Determine which agent just completed
            current_steps = list(state.keys())
            if current_steps:
                node_name = current_steps[0]
                node_state = state[node_name]
                
                # Extract step logs
                steps_log = node_state.get("steps_log", [])
                if steps_log and steps_log != last_step:
                    latest_log = steps_log[-1] if steps_log else ""
                    
                    # Parse the log to extract agent name and status
                    if "🛡️" in latest_log:
                        status_msg = "✅ Safety Check Complete"
                        next_msg = "🧭 Analyzing Intent..."
                    elif "🧭" in latest_log:
                        status_msg = "✅ Intent Classified"
                        next_msg = "✨ Refining Query..."
                    elif "✨" in latest_log:
                        status_msg = "✅ Query Refined"
                        next_msg = "💾 Executing Database Query..."
                    elif "💾" in latest_log or "🔍" in latest_log:
                        status_msg = "✅ Data Retrieved"
                        next_msg = "📝 Generating Response..."
                    elif "📝" in latest_log:
                        status_msg = "✅ Response Generated"
                        next_msg = None
                    else:
                        status_msg = latest_log[:50]
                        next_msg = None
                    
                    # Send completion of current step
                    yield f"data: {json.dumps({'type': 'step_complete', 'message': status_msg, 'log': latest_log})}\n\n"
                    
                    # Send start of next step
                    if next_msg:
                        yield f"data: {json.dumps({'type': 'status', 'message': next_msg})}\n\n"
                    
                    last_step = steps_log
                    
                    # Small delay to ensure UI updates
                    await asyncio.sleep(0.05)
        
        # Get final result
        result = app_graph.get_state(config).values
        
        total_execution_time = time.time() - request_start_time
        last_message = result['messages'][-1].content
        steps = result.get("steps_log", [])
        session_usage = result.get("token_usage_session", {"total": 0, "prompt": 0, "completion": 0})
        turn_usage = result.get("token_usage_turn", {"total": 0, "prompt": 0, "completion": 0})
        
        # Extract performance metrics
        performance_data = result.get("performance", {})
        performance_list = []
        
        for agent_name, metrics in performance_data.items():
            performance_list.append({
                "agent_name": agent_name,
                "total_time": metrics.get("total_time", 0),
                "llm_time": metrics.get("llm_time"),
                "db_time": metrics.get("db_time"),
                "search_time": metrics.get("search_time")
            })
        
        usage_display = {
            "turn": turn_usage,
            "session": session_usage
        }
        
        # Send final response
        final_data = {
            "type": "complete",
            "response": last_message,
            "steps": steps,
            "token_usage": usage_display,
            "performance": performance_list,
            "total_execution_time": total_execution_time
        }
        
        yield f"data: {json.dumps(final_data)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """
    Streaming endpoint using Server-Sent Events.
    """
    return StreamingResponse(
        generate_events(req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Original non-streaming endpoint for compatibility.
    """
    session_id = req.session_id
    config = {"configurable": {"thread_id": session_id}}
    request_start_time = time.time()
    
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
        
        usage_display = {
            "turn": turn_usage,
            "session": session_usage
        }
        
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