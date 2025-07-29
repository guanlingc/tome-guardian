from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import Annotated, TypedDict
from typing import Sequence


# defining a chatbot class
class GeminiChatBot:
    """
    GeminiChatBot provides an interactive chatbot interface using a Google Generative AI model.

    Attributes:
        llm: Instance of ChatGoogleGenerativeAI for generating responses.
        system_prompt: ChatPromptTemplate containing the system prompt and message placeholders.
        memory: MemorySaver instance for managing conversation state.
        config: Dictionary containing configuration options, including thread_id.
        app: Compiled workflow application for managing conversation flow.

    Args:
        model_name (str): Name of the generative AI model to use.
        system_prompt (str): System prompt to guide the chatbot's behavior.
        thread_id (str, optional): Identifier for the conversation thread. Defaults to "123".

    Methods:
        chat():
            Starts an interactive chat session with the user. Accepts user input, generates responses,
            and maintains conversation history. Type 'exit' to end the session.
    """
    def __init__(self, model_name, system_prompt, thread_id="123"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": thread_id}}
        self._build_workflow()

    def _build_workflow(self):
        class State(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
        def call_model(state: State):
            prompt = self.system_prompt.invoke(state)
            response = self.llm.invoke(prompt)
            return {"messages": response}
        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        self.llm = workflow.compile(checkpointer=self.memory)

    def chat(self):
        messages = []
        while True:
            user_text = input("User: ")
            if user_text.lower() == "exit":
                print("Goodbye!")
                break
            messages.append(HumanMessage(user_text))
            response = self.llm.invoke({"messages": messages}, config=self.config)
            ai_msg = response["messages"][-1]
            print(f"Tome Guardian: {ai_msg.content}")
            messages.append(ai_msg)
