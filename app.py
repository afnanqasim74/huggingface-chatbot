import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import os
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "db.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

@st.cache(allow_output_mutation=True)
def load_chain():
    model = T5ForConditionalGeneration.from_pretrained("google/t5-xxl-ssm-nq")
    tokenizer = T5Tokenizer.from_pretrained("google/t5-xxl-ssm-nq")
    chain = load_qa_chain(model, tokenizer, chain_type="stuff")
    return chain

def main():
    st.title("CHAT LUB")

    # Load model
    model = load_model()

    # Load chain
    chain = load_chain()

    # User input
    input_text = st.text_input("Enter the starting sentence")

    # Generate answer when button is clicked
    if st.button("ENTER"):
        progress_text = "Generating answer... 🤗"
        my_bar = st.progress(0)
        my_bar_text = st.empty()

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
            my_bar_text.text(f"{progress_text} {percent_complete + 1}%")

        # Perform similarity search
        # Replace this part with your own similarity search code

        # Generate answer using the chain
        generated_answer = chain.run(input_documents=docs, question=input_text)

        # Display the generated answer
        st.text_area("Generated answer", generated_answer, height=200)

if __name__ == "__main__":
    main()
