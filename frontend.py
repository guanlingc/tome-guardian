import gradio as gr
import os
import sys
from langchain_core.messages import HumanMessage

# Add the parent directory to the Python path to allow imports from 'api'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.chatbot import GeminiChatBot
from components.helper import get_env_variables

# Load environment variables
try:
    env_vars = get_env_variables()
    GOOGLE_API_KEY = env_vars["GOOGLE_API_KEY"]
    GEMINI_MODEL = env_vars["GEMINI_MODEL"]
except EnvironmentError as e:
    print(f"Error loading environment variables: {e}")
    print("Please ensure GOOGLE_API_KEY and GEMINI_MODEL are set in your .env file.")
    exit()

# Initialize the chatbot
system_prompt = "You are a helpful AI assistant. Respond to all questions."
chatbot_instance = GeminiChatBot(model_name=GEMINI_MODEL, system_prompt=system_prompt)

def respond(message, history):
    # The history parameter in Gradio's ChatInterface is a list of lists,
    # where each inner list is [user_message, bot_message].
    # We need to convert this into a format suitable for our GeminiChatBot,
    # which expects a sequence of BaseMessage objects.

    # For this simple integration, we'll just send the latest user message
    # to the chatbot and get a response. The GeminiChatBot internally
    # manages its own memory/state.
    
    # Note: The current GeminiChatBot implementation in chatbot.py uses
    # LangGraph's checkpointer for memory. When `invoke` is called, it
    # uses the `config` (which includes `thread_id`) to manage the conversation state.
    # So, we don't need to pass the full `history` from Gradio to `chatbot_instance.llm.invoke`.
    # We just need to pass the current message.
    user_input = HumanMessage(message)
    response = chatbot_instance.app.invoke({"messages": user_input}, config=chatbot_instance.config)
    ai_msg_content = response["messages"][-1].content
    return ai_msg_content

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Tome Guardian Chatbot",
    description="Greetings, What information do you require from the DaggerHearts Tome?",
    examples=[
        ["Tell me more about the daemon race."],
        ["What kind of information do you have about monsters?"],
        ["Tell me a fun fact about AI."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
