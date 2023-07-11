""""""
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FaissVectorStore
from langchain.huggingfacehub import HuggingFaceHub
import os

import streamlit as st
import time

import pickle

# Load the chain
@st.cache_resource
def model():
    load_path = "db.pkl"
    with open(load_path, "rb") as af:
        d = pickle.load(af)
        return d

def main():
    st.title("CHAT LUB")

    # Load the model
    d = model()

    # Load the chain
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 1, "max_length": 512})
    c = load_qa_chain(llm, chain_type="stuff")

    # User input
    input_text = st.text_input("Enter the starting sentence")

    # Generate answer when button is clicked
    if st.button("ENTER"):
        progress_text = "Generating answer...🤗"
        my_bar = st.progress(0)
        my_bar_text = st.empty()

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
            my_bar_text.text(f"{progress_text} {percent_complete + 1}%")

        # Perform similarity search
        vector = embeddings.encode([input_text])  # Assuming you have the embeddings object initialized
        k = 5  # The number of nearest neighbors to search for
        distances = faiss.IndexFlatIP().search(vector, k)
        indices = distances[1][0]
        docs = [d.get_document_by_id(doc_id) for doc_id in indices]

        # Generate answer using the chain
        generated_answer = c.run(input_documents=docs, question=input_text)

        # Display the generated answer
        st.text_area("Generated answer", generated_answer, height=200)

if __name__ == "__main__":
    main()
