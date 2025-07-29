# This script is just a dumping ground for obsolete functions
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def create_chat_memory():
    """
    create a temp dictionary storage. 
    assigns a session id as KEY in the dictionary to store the chat history as VALUE
    This is implemented as the memory module needs to be a object from BaseChatMessageHistory
    Usage:
        get_session_history = create_chat_memory()
        history = get_session_history(session_id)
    """
    
    chat_history_store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve the chat message history for a given session.
        If the session does not exist in the chat history store, a new in-memory chat message history is created and stored.
        Args:
            session_id (str): The unique identifier for the chat session.
        Returns:
            BaseChatMessageHistory: The chat message history associated with the session.
        """
        if session_id not in chat_history_store:
            chat_history_store[session_id] = InMemoryChatMessageHistory()
        return chat_history_store[session_id]
    return get_session_history