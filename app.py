import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain.huggingfacehub import HuggingFaceHub
import pickle
import os
import time

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
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 1, "max_length": 512})
    chain = load_qa_chain(llm, chain_type="stuff")
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
        index = faiss.IndexFlatIP(768)  # Assuming the vector size is 768, adjust if necessary
        index_path = "path_to_index"  # Replace with the actual path to the index
        index = faiss.read_index(index_path)
        k = 5  # The number of nearest neighbors to search for
        _, indices = index.search(embeddings.encode([input_text]), k)

        # Load documents from the model
        docs = [model.get_document_by_id(doc_id) for doc_id in indices.flatten()]

        # Generate answer using the chain
        generated_answer = chain.run(input_documents=docs, question=input_text)

        # Display the generated answer
        st.text_area("Generated answer", generated_answer, height=200)

if __name__ == "__main__":
    main()
