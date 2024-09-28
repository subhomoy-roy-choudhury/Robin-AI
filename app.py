import os
from abc import ABC, abstractmethod
import streamlit as st
from constants import AI_MODELS, PlatformEnum

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError
    

# Langchain
from langchain_core.prompts import ChatPromptTemplate

# Groq
from langchain_groq import ChatGroq

# HuggingFace Hub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class BaseLLM(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> str:
        pass

    @abstractmethod
    def format_messages(self, messages: list) -> str:
        pass


class GroqAILLM(BaseLLM):
    """
    Models List (Chat Completion) :=
    - https://console.groq.com/docs/models
    - https://console.groq.com/settings/limits
    """

    def __init__(self):
        pass

    def fetch_api_key(self) -> str:
        api_key = os.getenv("GROQ_API_KEY", None)
        if not api_key:
            raise Exception("Groq API Key Not Found")
        return api_key

    def load_model(self, model_name: str) -> str:
        groq_api_key = self.fetch_api_key()
        model = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
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


class HuggingFaceAILLM(BaseLLM):
    """
    HuggingFaceAILLM
    """

    def __init__(self):
        pass

    def fetch_api_key(self) -> str:
        api_key = os.getenv("HUGGINGFACE_API_KEY", None)
        if not api_key:
            raise Exception("HuggingFace API Key Not Found")
        return api_key

    def load_model(self, model_name: str) -> str:
        huggingface_api_key = self.fetch_api_key()
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=huggingface_api_key,
            repo_id=model_name,
            task="text-generation",
            repetition_penalty=1.03,
        )
        model = ChatHuggingFace(llm=llm)
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
    elif platform == PlatformEnum.HUGGINGFACE:
        return HuggingFaceAILLM()
    raise Exception("Error in fetching Model")


# Load configuration from file
def load_config(file_path: str = "auth.config.yaml") -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)


# Initialize authenticator
def initialize_authenticator(config: dict) -> stauth.Authenticate:
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["pre-authorized"],
    )


# Display authentication status and handle logout
def handle_authentication(authenticator: stauth.Authenticate):
    try:
        authenticator.login()
    except LoginError as e:
        st.error(e)

    if st.session_state.get("authentication_status"):
        authenticator.logout()
        st.write(f'Welcome *{st.session_state["name"]}*')
    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")


# Display sidebar options
def display_sidebar() -> dict:
    with st.sidebar:
        auto_select_model_check = st.checkbox("Select Model")

        model_index = st.selectbox(
            "Choose a model:",
            options=range(len(AI_MODELS)),  # Use index values as options
            disabled=not auto_select_model_check,
            format_func=lambda index: f"{AI_MODELS[index]['platform'].value.upper()}: {AI_MODELS[index]['name']}",
        )
        return {
            "auto_select_model_check": auto_select_model_check,
            "model_index": model_index,
        }


# Handle chat interaction
def handle_chat_interaction(auto_select_model_check: bool, model_information: dict):
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Generate response only if a model is selected and user has entered a prompt
        if auto_select_model_check and model_information:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    platform = model_information.get("platform")
                    model_func = LLM_Factory(platform)
                    model = model_func.load_model(model_information["model_name"])

                    # Generate assistant response
                    msg = model_func.chat(model, st.session_state.messages[1:])
                    if msg:
                        st.write(msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg}
                        )
        else:
            st.info("Please select a model to continue.")


# Main function to encapsulate all logic
def main():
    config = load_config()
    authenticator = initialize_authenticator(config)

    handle_authentication(authenticator)
    
    # Proceed only if user is authenticated
    if st.session_state.get("authentication_status"):
        sidebar_options = display_sidebar()
        auto_select_model_check = sidebar_options["auto_select_model_check"]
        model_information = AI_MODELS[sidebar_options["model_index"]]

        # Set main content title and description
        st.title("Robin AI")
        st.caption("🚀 A Streamlit chatbot powered by OpenAI")

        # Handle chat interaction
        handle_chat_interaction(auto_select_model_check, model_information)


# Run the main function
if __name__ == "__main__":
    main()
