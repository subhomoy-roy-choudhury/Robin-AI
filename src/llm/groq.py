import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llm.base import BaseLLM
from src.constants import UserRole

class GroqAILLM(BaseLLM):
    """
    Models List (Chat Completion) :=
    - https://console.groq.com/docs/models
    - https://console.groq.com/settings/limits
    """

    def __init__(self):
        super().__init__()

    def fetch_api_key(self, user_role: str = UserRole.MEMBER) -> str:
        api_key = None
        if user_role == UserRole.GUEST.value:
            api_key = st.text_input("Enter the Groq API key")
        elif user_role == UserRole.MEMBER.value:
            api_key = st.secrets.get("GROQ_API_KEY", None)
            
        if not api_key:
            st.error("Groq API Key Not Found")
        return api_key


    def load_model(self, model_name: str, user_role: str = UserRole.MEMBER):
        try:
            groq_api_key = self.fetch_api_key(user_role)
            model = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model_name=model_name,
            )
            self.model = model
        except Exception as _:
            st.error("Error in loading Groq model. Please check the API Key.")

    def format_messages(self, messages: list) -> str:
        return [
            (msg["role"] if msg["role"] != "user" else "human", msg["content"])
            for msg in messages
        ]

    def chat(self, messages: str) -> str:
        prompt = ChatPromptTemplate.from_messages(self.format_messages(messages))
        chain = prompt | self.model

        response = []
        for chunk in chain.stream({"input": ""}):
            response.append(chunk.content)
        return " ".join(response)
