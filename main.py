import streamlit as st
import numpy as np
import random
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pinecone  # Import pinecone to initialize it
from pinecone import Pinecone, Index
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
from github import Github, Repository
from git import Repo
from pathlib import Path
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore  # type: ignore
import dotenv
import json
import yfinance as yf
import concurrent.futures
import requests

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)

pinecone_api_key = st.secrets["PINECONE_API_KEY"]
os.environ['PINECONE_API_KEY'] = pinecone_api_key

index_name = "financial-analysis-llms"
namespace = "stock-descriptions"

hf_embeddings = HuggingFaceEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"],)

# Connect to your Pinecone index
pinecone_index = pc.Index(index_name)


def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=3, include_metadata=True, namespace=namespace)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are a expert stock market analyst with 30 years of experience and are tasked with responding to a newbies questions on the stock market.
    Provide concise, accurate, detailed, and researched explanations and insights for the user to be able to draw conclusions from.
    Always consider all of the context from previous messages when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

























st.title("Finance Bot")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3.1-70b-versatile"

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about the stock market..."):
    # display user message on site
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display bot message on site
    with st.chat_message("assistant"):
        # Use perform_rag instead of direct LLM call
        response = perform_rag(prompt)
        st.write(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})