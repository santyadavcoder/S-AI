import streamlit as st
from transformers import pipeline
from huggingface_hub import InferenceApi

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
api = InferenceApi(repo_id="facebook/bart-large-cnn", token="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")

# Function to generate summary using the pipeline
def generate_summary(chat_history):
    chat_text = " ".join(chat_history[-15:])  # Take the last 15 messages for context
    summary = summarizer(chat_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to generate summary using the Hugging Face API
def generate_summary_with_api(chat_text):
    return api(chat_text, parameters={"max_length": 150, "min_length": 50})

# Streamlit app configuration
st.set_page_config(
    page_title="S-AI",
    page_icon="👀",  # You can use an emoji or provide a path to an image
    layout="wide"
)

st.title("Generate Your Chat 💬 History Summary using S-AI")

# User input text area
user_input = st.text_area("Enter text to summarize:", height=200)

# Button to generate the summary
if st.button("Generate Summary"):
    if user_input:
        summary = generate_summary(user_input)
        st.write("### AI-Generated Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")
