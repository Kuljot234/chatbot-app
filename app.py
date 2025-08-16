import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

# ✅ Streamlit Page Config
st.set_page_config(page_title="NeuroChat Clone", page_icon="🤖", layout="wide")
st.title("🤖 NeuroChat Clone")

# ✅ Load Hugging Face Token from Streamlit Secrets
HF_API_KEY = st.secrets["HF_API_KEY"]


# ✅ Login to Hugging Face
login(token=HF_API_KEY)

# ✅ Initialize session for chat history
if "Messages" not in st.session_state:
    st.session_state.Messages = []

# ✅ Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "ibm-granite/granite-3.3-2b-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,   # safer for Streamlit Cloud (no GPU)
            token=HF_API_KEY
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ✅ Generate model response
def generate_response(prompt, model, tokenizer):
    formatted_prompt = f"Human: {prompt}\nAI:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("AI:")[-1].strip()

# ✅ Load the model once
with st.spinner("🚀 Loading model... Please wait."):
    model, tokenizer = load_model()
st.success("✅ Model loaded successfully!")

# ✅ Render chat history
for message in st.session_state.Messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Take user input and respond
if prompt := st.chat_input("💬 Ask something..."):
    st.session_state.Messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            response = generate_response(prompt, model, tokenizer)
            st.markdown(response)

    st.session_state.Messages.append({"role": "assistant", "content": response})


