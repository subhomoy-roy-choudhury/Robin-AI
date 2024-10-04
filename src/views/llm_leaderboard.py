import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import AI_MODELS

leaderboard_filename = "data/open-llm-leaderboard.json"
data = []

with open(leaderboard_filename, "r") as f:
    leaderboard_data = json.load(f)

leaderboard_models = [model["fullname"].lower() for model in leaderboard_data]
for model in AI_MODELS:
    if model["openllm_model_name"].lower() in leaderboard_models:
        matched_model_data = next(
            (item for item in leaderboard_data if item["fullname"].lower() == model["openllm_model_name"].lower()), None
        )
        if matched_model_data:
            data.append(matched_model_data)
    
df = pd.DataFrame(data)

def llm_list_page():
    st.title("LLM List")
    quick_filters = st.multiselect("Filters", df.columns.tolist())
    st.dataframe(df[quick_filters])
    
def llm_stats_page():
    st.title("LLM Stats")
    metric = st.selectbox("Metric", ["Average \u2b06\ufe0f", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO", "#Params (B)"])

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("LLM Stats"):
            show_llm_stats_as_table()

    with col2:
        if st.button("LLM List"):
            show_llm_list_as_table()

    # Select the top 10 models by the selected metric
    top_models = df.nlargest(columns=metric, n=10)[["Model", "Architecture", metric]]

    # Bar Chart using Matplotlib
    fig, ax = plt.subplots()
    ax.bar(top_models["Model"], top_models[metric], color='skyblue')
    ax.set_title(f"Top 10 Models by {metric} - Bar Chart")
    ax.set_xlabel("Model Name")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Pie Chart using Matplotlib
    fig, ax = plt.subplots()
    ax.pie(top_models[metric], labels=top_models["Model"], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    ax.set_title(f"Top 10 Models by {metric} - Pie Chart")
    st.pyplot(fig)

    # Scatter Chart using Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(top_models["Model"], top_models[metric], c='orange')
    ax.set_title(f"Top 10 Models by {metric} - Scatter Plot")
    ax.set_xlabel("Model Name")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
@st.dialog("LLM List - Stats")
def show_llm_stats_as_table():
    metric = st.selectbox("Metric ", ["Average \u2b06\ufe0f", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO", "#Params (B)"])
    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model", "Architecture", metric]]

    st.dataframe(top_models)

@st.dialog("LLM List ")
def show_llm_list_as_table():
    st.dataframe(df)

def llm_leaderboard_page():
    st.title("LLM Leaderboard")

    quick_filters = st.multiselect(
        "Filters ",
        [
            "Model",
            "Architecture",
            "Weight type",
            "Hub License",
        ],
        default=[
            "Model",
            "Architecture",
            "Weight type",
            "Hub License",
        ],
    )

    st.dataframe(df[quick_filters])

    metric = st.selectbox("Metric", ["Average \u2b06\ufe0f", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO", "#Params (B)"])

    # Top N Models

    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model", "Architecture", metric]]

    # Bar Chart using Matplotlib
    fig, ax = plt.subplots()
    ax.bar(top_models["Model"], top_models[metric], color='skyblue')
    ax.set_title(f"Top 10 Models by {metric} - Bar Chart")
    ax.set_xlabel("Model Name")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Pie Chart using Matplotlib
    fig, ax = plt.subplots()
    ax.pie(top_models[metric], labels=top_models["Model"], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    ax.set_title(f"Top 10 Models by {metric} - Pie Chart")
    st.pyplot(fig)

    # Scatter Chart using Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(top_models["Model"], top_models[metric], c='orange')
    ax.set_title(f"Top 10 Models by {metric} - Scatter Plot")
    ax.set_xlabel("Model Name")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    st.pyplot(fig)