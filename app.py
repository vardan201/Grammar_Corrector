import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch

# ✅ Set page configuration FIRST
st.set_page_config(page_title="Grammar Correction with FLAN-T5 LoRA", page_icon="📝")

# ------------------ 🧠 Load Model & Tokenizer ------------------ #
@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = PeftModel.from_pretrained(base_model, "./flan_t5_lora_model")
    tokenizer = AutoTokenizer.from_pretrained("./flan_t5_lora_model")
    return model.eval(), tokenizer

model, tokenizer = load_model()

# ------------------ 🧹 Grammar Correction Function ------------------ #
def correct_grammar(text):
    input_text = f"correct grammar: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------ 🎯 UI ------------------ #
st.title("📝 Grammar Correction App")
st.markdown("### Powered by LoRA-fine-tuned FLAN-T5")

user_input = st.text_area("Enter a grammatically incorrect sentence:", height=150)

if st.button("Correct Grammar"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        corrected = correct_grammar(user_input)
        st.success("✅ Corrected Sentence:")
        st.markdown(f"**{corrected}**")
