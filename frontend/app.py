import chainlit as cl
import requests
import uuid
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"

@cl.on_chat_start
async def start():
    """
    Initialize the session when a user opens the page.
    """
    # Generate a unique session ID for this conversation
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    
    # Welcome Message
    welcome_message = """
    # 🧾 Invoice Analytics Agent
    
    I can help you analyze your invoices, find items, and track expenses.
    
    ### 📝 Try asking:
    1. **"Total tax paid this year?"**
    2. **"Where did I buy the Samsung S24?"**
    3. **"Show me invoices from Amazon."**
    """
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Triggered when the user sends a message.
    """
    session_id = cl.user_session.get("session_id")
    
    # Create a message placeholder to show work in progress
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    # Prepare the payload
    payload = {
        "session_id": session_id,
        "message": message.content
    }
    
    try:
        # Call Backend API
        # We use cl.make_async to run the blocking requests call
        response = await cl.make_async(requests.post)(
            CHAT_ENDPOINT, 
            json=payload, 
            timeout=120 # 2 minute timeout for complex SQL generation
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_answer = data.get("response", "No response.")
            steps = data.get("steps", [])
            
            # 1. Visualize the Agent's Thinking Process
            # We create a parent step to hold the logs
            if steps:
                async with cl.Step(name="Agent Thinking Process") as parent_step:
                    for log in steps:
                        # Determine icon based on log content
                        icon = "🔧"
                        if "Planner" in log: icon = "🧠"
                        elif "SQL" in log: icon = "💾"
                        elif "Vector" in log: icon = "🔍"
                        elif "Guardrails" in log: icon = "🛡️"
                        elif "Summarizer" in log: icon = "📝"
                        
                        # Add child step for each log entry
                        async with cl.Step(name=f"{icon} Step", parent_id=parent_step.id) as child:
                            child.output = log
            
            # 2. Update the message with the final response
            msg.content = bot_answer
            await msg.update()
        else:
            # Handle API Errors
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail")
            except:
                error_detail = response.text
                
            msg.content = f"❌ **Error**: {error_detail}"
            await msg.update()
            
    except requests.exceptions.Timeout:
        msg.content = "⏱️ **Timeout**: The agent took too long to respond."
        await msg.update()
    except Exception as e:
        msg.content = f"💥 **System Error**: {str(e)}"
        await msg.update()