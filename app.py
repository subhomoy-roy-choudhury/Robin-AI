import os
from abc import ABC, abstractmethod
import streamlit as st
from constants import AI_MODELS, PlatformEnum

# Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class BaseLLM(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> str:
        pass

    @abstractmethod
    def format_messages(self, messages: list) -> str:
        pass


class GroqAILLM(BaseLLM):
    def __init__(self):
        self.groq_api_key = self.fetch_api_key()

    def fetch_api_key(self) -> str:
        api_key = os.getenv("GROQ_API_KEY", None)
        if not api_key:
            raise Exception("Groq API Key Not Found")
        return api_key

    def load_model(self, model_name: str) -> str:
        model = ChatGroq(
            temperature=0,
            groq_api_key=self.groq_api_key,
            model_name=model_name,
        )
        return model

    def format_messages(self, messages: list) -> str:
        return [
            (msg["role"] if msg["role"] != "user" else "human", msg["content"])
            for msg in messages
        ]

    def chat(self, model, messages: str) -> str:
        prompt = ChatPromptTemplate.from_messages(self.format_messages(messages))
        chain = prompt | model

        response = []
        for chunk in chain.stream({"input": ""}):
            response.append(chunk.content)
        return " ".join(response)


def LLM_Factory(platform: str) -> BaseLLM:
    if platform == PlatformEnum.GROQ:
        return GroqAILLM()
    raise Exception("Error in fetching Model")


if __name__ == "__main__":
    auto_select_model_check = False

    # SideBar
    with st.sidebar:
        auto_select_model_check = st.checkbox("Select Model")
        
        model_index = st.selectbox(
            "Choose a model:",
            options=range(len(AI_MODELS)),  # Use index values as options
            disabled=not auto_select_model_check,
            format_func=lambda index: f"{AI_MODELS[index]['platform'].value.upper()}: {AI_MODELS[index]['name']}",
        )
        model_information = AI_MODELS[model_index]

    # Title
    st.title("Robin AI")

    st.caption("🚀 A Streamlit chatbot powered by OpenAI")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        msg = None

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if auto_select_model_check:
                        platform = model_information.get("platform")
                        model_func = LLM_Factory(platform)
                        model = model_func.load_model(model_information["model_name"])

                        msg = model_func.chat(model, st.session_state.messages[1:])
                        st.write(msg)
                    else:
                        st.info("Please select a model to continue.")
                        st.stop()

            # Append response only if msg is not None
            if msg:
                st.session_state.messages.append({"role": "assistant", "content": msg})
