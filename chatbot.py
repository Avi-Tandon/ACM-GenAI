import os
import streamlit as st
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import traceback

# Set the Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = "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"

# Cache the model to avoid reloading on every request
@st.cache_resource
def load_model(model_name):
    # Load the model with the provided API token
    return HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

# Refined prompt template
prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
    You are a helpful and conversational AI assistant. You have been having the following conversation with a user:
    
    Conversation history:
    {history}
    
    The user just asked: {question}
    
    Respond in a friendly, helpful way:
    """
)

# Function to trim conversation history to avoid exceeding model input length
def trim_conversation_history(history, max_tokens=300):
    # Split conversation history into lines (messages)
    messages = history.split("\n")
    
    # Keep only the last few messages to fit within token limit
    while len("\n".join(messages)) > max_tokens:
        messages.pop(0)
    
    return "\n".join(messages)

# Streamlit UI setup
st.title("Conversational AI Chatbot")
st.write("Built using Hugging Face and LangChain")

# Model selection
model_name = st.selectbox(
    "Choose a model",
    ("facebook/blenderbot-400M-distill", "google/flan-t5-base", "gpt2")
)

st.write(f"You selected the model: {model_name}")

# Load the selected model
llm = load_model(model_name)

# Initialize LangChain's LLMChain to handle conversation history
chatbot = LLMChain(llm=llm, prompt=prompt)

# Keep track of conversation history across messages
if "history" not in st.session_state:
    st.session_state["history"] = ""

# User input area for conversation
user_input = st.text_input("You: ")

# Process the user's input and generate a response
if user_input:
    # Trim conversation history to avoid exceeding input length
    st.session_state["history"] = trim_conversation_history(st.session_state["history"])

    # Append the user's input to conversation history
    st.session_state["history"] += f"User: {user_input}\n"
    
    # Generate response using the chatbot
    try:
        response = chatbot.run({
            "history": st.session_state["history"],
            "question": user_input
        })
    except Exception as e:
        # Print the full error stack trace for debugging
        st.write("Error occurred:")
        st.write(traceback.format_exc())
        response = "Sorry, I am unable to respond at the moment."
    
    # Append the chatbot's response to conversation history
    st.session_state["history"] += f"AI: {response}\n"
    
    # Display the full conversation history
    st.write(st.session_state["history"])
