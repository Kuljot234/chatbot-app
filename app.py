import streamlit as st
from huggingface_hub import InferenceClient

# Load Hugging Face API key securely from Streamlit secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize client
client = InferenceClient(api_key=HF_API_KEY)

st.title("🤖 Hugging Face Chatbot with Memory")

# Initialize chat history in Streamlit session_state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Say something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from Hugging Face
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",  # public free model
            messages=st.session_state.messages,
            max_tokens=500,
        )

        ai_reply = response.choices[0].message["content"]

        # Add AI response
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        with st.chat_message("assistant"):
            st.markdown(ai_reply)

    except Exception as e:
        st.error(f"Error: {e}")
