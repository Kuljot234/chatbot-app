import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

# ✅ Streamlit Page Config
st.set_page_config(page_title="Chat Bot", page_icon="🤖")
st.title("🤖 Chat Bot")

# ✅ Your Hugging Face Token
HF_API_KEY = "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"

# ✅ Login to Hugging Face if token is set
if HF_API_KEY and HF_API_KEY != "your_huggingface_api_key_here":
    login(token=HF_API_KEY)

# ✅ Initialize session for chat history
if "Messages" not in st.session_state:
    st.session_state.Messages = []

# ✅ Load model and tokenizer
@st.cache_resource
def load_model():
    if HF_API_KEY == "your_huggingface_api_key_here":
        st.error("Please set your Hugging Face API key in the code.")
        st.stop()
    model_name = "ibm-granite/granite-3.3-2b-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            token=HF_API_KEY
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ✅ Generate model response
def generate_response(prompt, model, tokenizer):
    formatted_prompt = f"Human: {prompt}\nAI:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("AI:")[-1].strip()
    return response

# ✅ Load the model with Streamlit spinner
with st.spinner("Loading model..."):
    model, tokenizer = load_model()
st.success("Model loaded successfully!")

# ✅ Render chat history
for message in st.session_state.Messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Take user input and respond
if prompt := st.chat_input("Ask your question:"):
    st.session_state.Messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(prompt, model, tokenizer)
            st.markdown(response)

    st.session_state.Messages.append({"role": "assistant", "content": response})
