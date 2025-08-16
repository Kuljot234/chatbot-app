import streamlit as st
from huggingface_hub import InferenceClient

# Load your Hugging Face API key from Streamlit secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize Hugging Face Inference Client
client = InferenceClient(token=HF_API_KEY)

st.title("🤖 Hugging Face Chatbot")

# User input
user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    try:
        # Use a public model (free, no license needed)
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # ✅ Public safe model
            messages=[{"role": "user", "content": user_input}],
            max_tokens=200
        )

        # Display chatbot response
        st.write("**Bot:**", response.choices[0].message["content"])

    except Exception as e:
        st.error(f"Error: {e}")
