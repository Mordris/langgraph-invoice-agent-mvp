import chainlit as cl
import requests
import uuid
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    await cl.Message(content="# 🧾 Invoice Agent Ready").send()

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    payload = {"session_id": session_id, "message": message.content}
    
    try:
        response = await cl.make_async(requests.post)(
            CHAT_ENDPOINT, json=payload, timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_answer = data.get("response", "")
            steps = data.get("steps", [])
            usage = data.get("token_usage", {})
            
            # 1. Render Steps
            if steps:
                async with cl.Step(name="Agent Thinking") as parent:
                    for log in steps:
                        icon = "🔧"
                        if "Planner" in log: icon = "🧠"
                        elif "SQL" in log: icon = "💾"
                        elif "Vector" in log: icon = "🔍"
                        elif "Guardrails" in log: icon = "🛡️"
                        elif "Summarizer" in log: icon = "📝"
                        
                        async with cl.Step(name=f"{icon} Step", parent_id=parent.id) as child:
                            child.output = log
            
            # 2. Append Token Usage
            if usage:
                turn = usage.get('turn', {})
                sess = usage.get('session', {})
                
                token_info = f"""
                
---
**📊 Token Usage**
*   **This Turn**: {turn.get('total', 0)}
*   **Session Total**: {sess.get('total', 0)}
"""
                bot_answer += token_info

            msg.content = bot_answer
            await msg.update()
        else:
            msg.content = f"❌ Error: {response.text}"
            await msg.update()
            
    except Exception as e:
        msg.content = f"💥 Error: {str(e)}"
        await msg.update()