
""""""
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import os

import streamlit as st
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
import time





import os

@st.cache_resource
def libraries():

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})


    c = load_qa_chain(llm, chain_type="stuff")

    embeddings = HuggingFaceEmbeddings()



    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    return c 

c = libraries()

# Text Splitter

# Load the chain
import pickle




# Load the chain
@st.cache_resource
def model():

    load_path = "db.pkl"
    with open(load_path, "rb") as af:
        d = pickle.load(af)
        return d
d  = model()


def main():
    st.title("CHAT LUB ")

    # User input
    input_text = st.text_input("Enter the starting sentence")

    # Generate story when button is clicked
    if st.button("ENTER"):
        progress_text = "Generating answer...🤗"
        my_bar = st.progress(0)
        my_bar_text = st.empty()

        for percent_complete in range(100):
            time.sleep(.01)
            my_bar.progress(percent_complete + 1)
            my_bar_text.text(f"{progress_text} {percent_complete + 1}%")
        docs = d.similarity_search(input_text)
        a = c.run(input_documents=docs, question=input_text) 
        generated_story = a

        # Display the generated story
        #st.write("Generated Story:")
        st.text_area("Generated anwer", generated_story, height=200)

if __name__ == "__main__":
    main()
