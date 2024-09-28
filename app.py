import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator import LoginError


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
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout()
    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")


# --- MAIN APP ---
def main():
    # Load authentication configuration
    config = load_config()

    # Initialize authenticator
    authenticator = initialize_authenticator(config)

    # Handle authentication
    handle_authentication(authenticator)

    # Check if user is authenticated before displaying the app
    if st.session_state.get("authentication_status"):
        # --- PAGE SETUP ---
        chatbot_page = st.Page(
            "views/robinai_chatbot.py",
            title="Robin AI",
            icon=":material/account_circle:",
            default=True,
        )
        gorilla_app = st.Page(
            "views/gorilla_llm.py",
            title="Gorilla LLM",
            icon=":material/bar_chart:",
        )
        llm_optimiser = st.Page(
            "views/llm_cost_optimiser.py",
            title="LLM Cost Optimiser",
            icon=":material/bar_chart:",
        )

        pg = st.navigation(
            {
                "Apps": [chatbot_page, gorilla_app],
                "Miscelleneous": [llm_optimiser],
            }
        )

        # --- SHARED ON ALL PAGES ---
        st.sidebar.markdown(
            "Made with ❤️ by [Subhomoy Roy Choudhury](https://github.com/subhomoy-roy-choudhury)"
        )

        # --- RUN NAVIGATION ---
        pg.run()
    else:
        st.warning("You need to login to access the app.")


if __name__ == "__main__":
    main()
