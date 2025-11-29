import chainlit as cl
import requests
import uuid
import os
import asyncio

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    await cl.Message(content="# 🧾 Invoice Agent Ready\n\nAsk me anything about your invoices!").send()

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    
    # Create a message with initial status
    msg = cl.Message(content="")
    await msg.send()
    
    # Create a step to show real-time progress
    async with cl.Step(name="🤖 Agent Process", type="tool") as process_step:
        
        # Step 1: Guardrails
        status_step = await cl.Step(name="🛡️ Checking Safety...", parent_id=process_step.id).send()
        
        payload = {"session_id": session_id, "message": message.content}
        
        try:
            # Make async request
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_answer = data.get("response", "")
                steps = data.get("steps", [])
                usage = data.get("token_usage", {})
                performance = data.get("performance", [])
                total_time = data.get("total_execution_time", 0)
                
                # Update status with completion
                status_step.name = "✅ Safety Check Complete"
                await status_step.update()
                
                # Display each step with professional naming and timing
                step_icons = {
                    "Guardrails": ("🛡️", "Safety Check"),
                    "Intent": ("🧭", "Intent Classification"),
                    "Refiner": ("✨", "Query Refinement"),
                    "SQL": ("💾", "Database Query"),
                    "Vector": ("🔍", "Semantic Search"),
                    "Summarizer": ("📝", "Response Generation")
                }
                
                for log in steps:
                    # Extract agent name from log
                    agent_name = None
                    for key in step_icons.keys():
                        if key in log:
                            agent_name = key
                            break
                    
                    if agent_name:
                        icon, display_name = step_icons[agent_name]
                        # Extract timing if available
                        timing = ""
                        if "(" in log and "s)" in log:
                            timing = log[log.find("("):log.find("s)")+2]
                        
                        step_name = f"{icon} {display_name} {timing}"
                    else:
                        step_name = log[:50]
                    
                    async with cl.Step(name=step_name, parent_id=process_step.id) as agent_step:
                        agent_step.output = log
                
                # Display performance metrics in a structured way
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
                    
                    async with cl.Step(name="⚡ Performance Metrics", parent_id=process_step.id) as perf_step:
                        perf_step.output = perf_text
                
                # Display token usage
                if usage:
                    turn = usage.get('turn', {})
                    sess = usage.get('session', {})
                    
                    # Calculate approximate cost (GPT-4o-mini pricing)
                    input_cost = (turn.get('prompt', 0) / 1_000_000) * 0.150  # $0.150 per 1M input tokens
                    output_cost = (turn.get('completion', 0) / 1_000_000) * 0.600  # $0.600 per 1M output tokens
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
                    
                    async with cl.Step(name="💰 Cost Analysis", parent_id=process_step.id) as cost_step:
                        cost_step.output = token_text
                
                # Update main message with final response
                msg.content = bot_answer
                await msg.update()
                
            else:
                msg.content = f"❌ Error: {response.text}"
                await msg.update()
                
        except Exception as e:
            msg.content = f"💥 Error: {str(e)}"
            await msg.update()