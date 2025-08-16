import streamlit as st
from huggingface_hub import InferenceClient

# Load Hugging Face API Key
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key=HF_API_KEY)

# Streamlit UI
st.title("🤖 Hugging Face Chatbot")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if user_input := st.chat_input("Type your message..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.text_generation(
                    model="meta-llama/Llama-2-7b-chat-hf",  # You can swap models
                    prompt=user_input,
                    max_new_tokens=300,
                    temperature=0.7
                )
                st.markdown(response)
                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
