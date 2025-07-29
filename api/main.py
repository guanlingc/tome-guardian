import os
from dotenv import load_dotenv

from typing import Sequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# import helper functions
import components.helper as helper
from components.chatbot import GeminiChatBot

# define the state class
# class State(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]



    
    
# def main():
#     # load environment variables (ie google gemini key)
#     env= helper.get_env_variables()
#     # define llm model and chat conditions
#     llm = ChatGoogleGenerativeAI(model=env["GEMINI_MODEL"], temperature = 0.3)
#     # define system behaviour of LLM, messages placeholder to contain the chat history
#     system_prompt = ChatPromptTemplate.from_messages([
#         ("system","You are a helpful assistant"),
#         MessagesPlaceholder(variable_name="messages")
#     ])
#     # function for to call LLM    
#     def call_model(state: State):
#         prompt = system_prompt.invoke(state)
#         response = llm.invoke(prompt)
#         return {"messages": response}
#     # create the workflow graph with a Start Node and connect to a model node that calls LLM
#     workflow = StateGraph(state_schema=State)
#     workflow.add_edge(START, "model")
#     workflow.add_node("model", call_model)
#     # creates a memory function for LLM 
#     memory = MemorySaver()
#     app = workflow.compile(checkpointer=memory)
#     config = {"configurable": {"thread_id": "123"}}
#     # while loop to keep chatting
#     while True:
#         # get user input 
#         user_text = input("User: ")
#         if user_text == "exit":
#             print("Goodbye!")
#             break
#         # pass user input to LLM as the prompt
#         response = helper.chat_w_llm(user_text, app, config)
#         print(f"Gemini: {response}")


def main():
    env = helper.get_env_variables()
    system_prompt = "You are a helpful assistant"
    llm =  GeminiChatBot(
        model_name=env["GEMINI_MODEL"], 
        system_prompt= system_prompt,
        thread_id="123")
    llm.chat()

# runs script only when it is executed directly
if __name__ == "__main__":
    main()
