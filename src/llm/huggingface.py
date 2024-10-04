import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from llm.base import BaseLLM
from src.constants import UserRole

class HuggingFaceAILLM(BaseLLM):
    """
    HuggingFaceAILLM
    """

    def __init__(self):
        super().__init__()

    def fetch_api_key(self, user_role: str = UserRole.MEMBER) -> str:
        api_key = None
        if user_role == UserRole.GUEST.value:
            api_key = st.text_input("Enter the Huggingface API key")
        elif user_role == UserRole.MEMBER.value:
            api_key = st.secrets.get("HUGGINGFACE_API_KEY", None)
            
        if not api_key:
            st.error("HuggingFace API Key Not Found")
        return api_key

    def load_model(self, model_name: str, user_role: str = UserRole.MEMBER) -> str:
        try:
            huggingface_api_key = self.fetch_api_key(user_role)
            llm = HuggingFaceEndpoint(
                huggingfacehub_api_token=huggingface_api_key,
                repo_id=model_name,
                task="text-generation",
                repetition_penalty=1.03,
            )
            model = ChatHuggingFace(llm=llm)
            self.model = model
            return True
        except Exception as _:
            st.error("Error in loading Huggingface model. Please check the API Key.")
        return False
    
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
