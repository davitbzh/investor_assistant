import os
from dotenv import load_dotenv

import hopsworks
import streamlit as st

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from functions.prompt_engineering import get_reranker, get_context_and_source, get_answer_from_gemini, get_answer_from_gpt

import config
import warnings
warnings.filterwarnings('ignore')


# Load the .env file
load_dotenv()

st.title("💬 AI assistant")

# Define a global variable for the OpenAI client
openai_client = None

@st.cache_resource()
def connect_to_hopsworks():
    # Initialize Hopsworks feature store connection
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Retrieve the 'documents' feature view
    stanford_reports_view = fs.get_feature_view(
        name="stanford_reports",
        version=1)
    
    eqt_portfolio_view = fs.get_feature_view(
        name="eqt_portfolio",
        version=1)

    return stanford_reports_view, eqt_portfolio_view


@st.cache_resource()
def get_models(saved_model_dir):

    # Load the Sentence Transformer
    sentence_transformer = SentenceTransformer(
        config.MODEL_SENTENCE_TRANSFORMER,
    ).to(config.DEVICE)

    reranker = get_reranker(config.RERANKER)
    
    return sentence_transformer, reranker

def set_openai_client():
    """Initialize and set the OpenAI client globally."""
    global openai_client
    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

def predict(user_query, sentence_transformer, feature_views, reranker, model):
    st.write('⚙️ Generating Response...')
    
    stanford_reports_view = feature_views[0]
    eqt_portfolio_view = feature_views[1]
    
    # Retrieve reranked context and source
    reports_and_source = get_context_and_source(user_query=user_query, 
                                                sentence_transformer=sentence_transformer, 
                                                feature_view=stanford_reports_view, 
                                                reranker=reranker,
                                                year=2024, 
                                                k=50)
    companies_and_source = get_context_and_source(user_query=user_query, 
                                                  sentence_transformer=sentence_transformer, 
                                                  feature_view=eqt_portfolio_view, 
                                                  reranker=reranker)

    reports_company_context = reports_and_source[0] + companies_and_source[0]

    # Generate model response
    if model == "GPT":
        if openai_client is None:
            st.error("OpenAI client is not initialized. Please check the API key.")
            return "OpenAI client error."
        return get_answer_from_gpt(query=user_query, context=reports_company_context, source=reports_and_source[1], 
                                   gpt_model=config.GPT_MODEL, client=openai_client)        
    elif model == "Gemini":
        return get_answer_from_gemini(query=user_query, context=reports_company_context, source=reports_and_source[1], api_key=os.environ["GEMINI_KEY"])
    else:
        return "Unknown model. Please select GPT or Gemini."


# Retrieve the feature view and the saved_model_dir
feature_views = connect_to_hopsworks()

# Load and retrieve the sentence_transformer and reranker
sentence_transformer, reranker = get_models(saved_model_dir=None)

# Model selection
model_choice = st.radio(
    "Select the model you want to use:",
    options=["GPT", "Gemini"],
    index=0,  # Default selection is the first option
    horizontal=True
)

# Set the OpenAI client globally if GPT is selected
if model_choice == "GPT" and openai_client is None:
    set_openai_client()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    response = predict(user_query=user_query, 
                       sentence_transformer=sentence_transformer, 
                       feature_views=feature_views, 
                       reranker=reranker,
                       model=model_choice)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
