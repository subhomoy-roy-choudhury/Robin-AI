from abc import ABC, abstractmethod
import streamlit as st
from src.constants import AI_MODELS
from src.llm import LLM_Factory, BaseLLM

# Display sidebar options
def display_sidebar(role : str = "member") -> dict:
    with st.sidebar:
        model_index = st.selectbox(
            "Choose a model:",
            options=range(len(AI_MODELS)),  # Use index values as options
            disabled=False,
            format_func=lambda index: f"{AI_MODELS[index]['platform'].value.upper()}: {AI_MODELS[index]['name']}",
        )
        
        # Fetch Model Information
        model_information = AI_MODELS[model_index]
        platform = model_information.get("platform")
        model = LLM_Factory(platform.name)
        loaded = model.load_model(model_information["model_name"], role)
        
        return {
            "model": model if loaded else None,
        }


# Handle chat interaction
def handle_chat_interaction(model: BaseLLM):
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
        if model:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    # Generate assistant response
                    msg = model.chat(st.session_state.messages[1:])
                    if msg:
                        st.write(msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg}
                        )
        else:
            st.info("Please select a model to continue or check the API key")


# Main Function for Chatbot
if st.session_state.get("authentication_status"):
    # Check Role
    role = st.session_state.role
    
    sidebar_options = display_sidebar(role)
    
    # Set main content title and description
    st.title("Robin AI")
    st.caption("🚀 A Streamlit chatbot powered by Multiple LLMs")

    # Handle chat interaction
    handle_chat_interaction(sidebar_options["model"])
