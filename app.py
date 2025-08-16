import streamlit as st
from huggingface_hub import InferenceClient

# Use your Hugging Face token securely from secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize client
client = InferenceClient(token=HF_API_KEY)

st.title("🤖 Hugging Face Chatbot")

# Input box for user
user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    try:
        # Use a Hugging Face chat model (LLaMA-3, Mistral, etc.)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",  # Change if you want another model
            messages=[{"role": "user", "content": user_input}],
            max_tokens=200
        )

        # Display response
        st.write("**Bot:**", response.choices[0].message["content"])

    except Exception as e:
        st.error(f"Error: {e}")
