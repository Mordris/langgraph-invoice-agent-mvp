import chainlit as cl
import uuid
import os
import httpx
import json

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"

@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    await cl.Message(content="# 🧾 Invoice Agent Ready\n\nAsk me anything about your invoices!").send()

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    
    # Create main message that will be updated
    msg = cl.Message(content="")
    await msg.send()
    
    # Create parent step for all agent processes
    agent_process_step = None
    current_status_step = None
    completed_steps = []
    
    try:
        # Use httpx for async streaming
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {"session_id": session_id, "message": message.content}
            
            async with client.stream("POST", STREAM_ENDPOINT, json=payload) as response:
                if response.status_code != 200:
                    msg.content = f"❌ Error: {response.status_code}"
                    await msg.update()
                    return
                
                # Create parent step on first event
                if agent_process_step is None:
                    agent_process_step = cl.Step(name="🤖 Agent Process", type="tool")
                    await agent_process_step.send()
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        event_type = data.get("type")
                        
                        if event_type == "status":
                            # Update current status (in-progress agent)
                            status_msg = data.get("message", "Processing...")
                            
                            if current_status_step:
                                # Update existing step to show completion
                                await current_status_step.remove()
                            
                            # Create new status step
                            current_status_step = cl.Step(
                                name=status_msg,
                                parent_id=agent_process_step.id,
                                type="tool"
                            )
                            await current_status_step.send()
                        
                        elif event_type == "step_complete":
                            # Agent completed - update status step to show completion
                            completion_msg = data.get("message", "Complete")
                            full_log = data.get("log", "")
                            
                            if current_status_step:
                                current_status_step.name = completion_msg
                                current_status_step.output = full_log
                                await current_status_step.update()
                                
                                # Store for later reference
                                completed_steps.append({
                                    "name": completion_msg,
                                    "log": full_log
                                })
                                
                                current_status_step = None
                        
                        elif event_type == "complete":
                            # Final response received
                            bot_answer = data.get("response", "")
                            steps = data.get("steps", [])
                            usage = data.get("token_usage", {})
                            performance = data.get("performance", [])
                            total_time = data.get("total_execution_time", 0)
                            
                            # Remove any lingering status step
                            if current_status_step:
                                await current_status_step.remove()
                            
                            # Display performance metrics
                            if performance:
                                perf_text = "**Performance Breakdown:**\n\n"
                                total_llm_time = 0
                                total_db_time = 0
                                
                                for perf in performance:
                                    agent = perf.get("agent_name", "Unknown")
                                    total = perf.get("total_time", 0)
                                    llm = perf.get("llm_time", 0)
                                    db = perf.get("db_time", 0)
                                    
                                    if llm:
                                        total_llm_time += llm
                                    if db:
                                        total_db_time += db
                                    
                                    perf_text += f"• **{agent.capitalize()}**: {total:.3f}s"
                                    if llm:
                                        perf_text += f" (LLM: {llm:.3f}s)"
                                    if db:
                                        perf_text += f" (DB: {db:.3f}s)"
                                    perf_text += "\n"
                                
                                perf_text += f"\n**Total Execution**: {total_time:.3f}s\n"
                                if total_llm_time > 0:
                                    perf_text += f"**Total LLM Time**: {total_llm_time:.3f}s ({(total_llm_time/total_time*100):.1f}%)\n"
                                if total_db_time > 0:
                                    perf_text += f"**Total DB Time**: {total_db_time:.3f}s ({(total_db_time/total_time*100):.1f}%)\n"
                                
                                perf_step = cl.Step(
                                    name="⚡ Performance Metrics",
                                    parent_id=agent_process_step.id
                                )
                                perf_step.output = perf_text
                                await perf_step.send()
                            
                            # Display token usage
                            if usage:
                                turn = usage.get('turn', {})
                                sess = usage.get('session', {})
                                
                                # Calculate costs (GPT-4o-mini pricing)
                                input_cost = (turn.get('prompt', 0) / 1_000_000) * 0.150
                                output_cost = (turn.get('completion', 0) / 1_000_000) * 0.600
                                turn_cost = input_cost + output_cost
                                
                                session_input_cost = (sess.get('prompt', 0) / 1_000_000) * 0.150
                                session_output_cost = (sess.get('completion', 0) / 1_000_000) * 0.600
                                session_cost = session_input_cost + session_output_cost
                                
                                token_text = f"""**Token Usage:**
• **This Turn**: {turn.get('total', 0):,} tokens (≈${turn_cost:.4f})
  - Input: {turn.get('prompt', 0):,}
  - Output: {turn.get('completion', 0):,}

• **Session Total**: {sess.get('total', 0):,} tokens (≈${session_cost:.4f})
  - Input: {sess.get('prompt', 0):,}
  - Output: {sess.get('completion', 0):,}"""
                                
                                cost_step = cl.Step(
                                    name="💰 Cost Analysis",
                                    parent_id=agent_process_step.id
                                )
                                cost_step.output = token_text
                                await cost_step.send()
                            
                            # Update main message with final response
                            msg.content = bot_answer
                            await msg.update()
                            
                            # Mark agent process as complete
                            if agent_process_step:
                                agent_process_step.name = "✅ Agent Process Complete"
                                await agent_process_step.update()
                        
                        elif event_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            msg.content = f"💥 Error: {error_msg}"
                            await msg.update()
                            
                            if agent_process_step:
                                agent_process_step.name = "❌ Agent Process Failed"
                                await agent_process_step.update()
                            
                            break
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing event: {e}")
                        continue
    
    except Exception as e:
        msg.content = f"💥 Connection Error: {str(e)}"
        await msg.update()
        
        if agent_process_step:
            agent_process_step.name = "❌ Connection Failed"
            await agent_process_step.update()