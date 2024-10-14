import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator import LoginError
from helpers import load_config
from views.llm_leaderboard import llm_list_page, llm_stats_page, llm_leaderboard_page


# Initialize authenticator
def initialize_authenticator(config: dict) -> stauth.Authenticate:
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


import streamlit as st
import streamlit_authenticator as stauth


def guest_login_flow():
    """Handle the guest login flow."""
    if st.button("Signup as Guest"):
        st.session_state.update(
            {
                "guest_clicked": True,
                "authentication_status": True,
                "role": "guest",
                "name": "Guest",
            }
        )
        return True  # Indicates that the user has signed up as a guest
    return False  # Guest login has not been initiated


def member_login_flow(authenticator: stauth.Authenticate):
    """Handle the member login flow using the authenticator."""
    try:
        authenticator.login()
    except LoginError as e:
        st.error(e)

    # Display appropriate message based on authentication status
    if st.session_state.get("authentication_status"):
        st.write(f"Welcome, *{st.session_state['name']}*")
        authenticator.logout("Logout", "sidebar")
        st.session_state.role = "member"
    elif st.session_state.get("authentication_status") is False:
        st.error("Incorrect username or password.")
    else:
        st.warning("Please enter your username and password.")


def handle_authentication(authenticator: stauth.Authenticate):
    """Manage the overall authentication flow based on guest or member login."""
    # Separate flow for guest login
    if st.session_state.get("guest_clicked", False):
        st.write("You are logged in as Guest")
        st.write(f"Welcome, *{st.session_state['name']}*")
        st.session_state.role = "guest"
    else:
        # Attempt guest login first
        guest_signed_up = guest_login_flow()

        # If not a guest, proceed with member login
        if not guest_signed_up:
            member_login_flow(authenticator)


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
        llm_list = st.Page(
            llm_list_page, title="LLM List", icon=":material/bar_chart:"
        )
        llm_stats = st.Page(
            llm_stats_page, title="LLM Stats", icon=":material/bar_chart:"
        )
        llm_leaderboard = st.Page(
            llm_leaderboard_page, title="LLM leaderboard", icon=":material/bar_chart:"
        )

        pg = st.navigation(
            {
                "Apps": [chatbot_page, gorilla_app, llm_optimiser],
                "Miscelleneous": [llm_list, llm_stats, llm_leaderboard],
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
