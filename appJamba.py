import streamlit as st
import PyPDF2
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
import openai
import nltk
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
from textblob import TextBlob
import spacy
import asyncio
import os
import ai21
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat


nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Load SpaCy model from local directory
model_path = os.path.join(os.path.dirname(__file__), 'en_core_web_sm/en_core_web_sm-3.6.0')
nlp = spacy.load(model_path)

# Set your OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]
ai21_api_key = st.secrets["ai21"]["api_key"]
client = AI21Client(api_key=ai21_api_key)
system_prompt = (
                        "You are a helpful and knowledgeable assistant. You are given a set of text chunks from documents, along with metadata such as title, summary, and keywords. "
                        "Please find the most relevant information based on the question below, "
                        "using only the provided chunks and metadata. Ensure your response is comprehensive, accurate, and informative, "
                        "covering all aspects of the question to the best of your ability."
                    )
# Streamlit app setup
st.title("Conversational Document Query App using Jamba 1.5")

def read_pdf(file):
    """Read a PDF file and convert it to text."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def convert_docx_to_text(file):
    """Convert a DOCX file to text."""
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def process_documents(uploaded_files):
    """Process PDF and DOCX files."""
    documents = []

    # Process uploaded files
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = read_pdf(file)
            documents.append(text)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = convert_docx_to_text(file)
            documents.append(text)

    return documents

def generate_title(chunk):
    return chunk.split('.')[0][:50] + '...'

def extract_keywords(chunk):
    r = Rake()
    r.extract_keywords_from_text(chunk)
    return r.get_ranked_phrases()

def generate_summary(chunk):
    sentences = sent_tokenize(chunk)
    return sentences[0] if len(sentences) > 1 else chunk[:100] + '...'

def extract_entities(chunk):
    doc = nlp(chunk)
    return [(ent.text, ent.label_) for ent in doc.ents]

def generate_questions(chunk):
    # Basic example, could be enhanced with a language model
    return ["What is this chunk about?", "What key points are discussed?"]

def augment_chunk(chunk):
    return {
        "chunk": chunk,
        "title": generate_title(chunk),
        "keywords": extract_keywords(chunk)
    }

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_embeddings_and_store(documents):
    """Create embeddings for the documents and store them in FAISS."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Split documents into smaller chunks using NLTK
    all_chunks = []
    for document in documents:
        chunks = split_text_into_chunks(document, chunk_size=500)
        for chunk in chunks:
            augmented_chunk = augment_chunk(chunk)
            all_chunks.append(augmented_chunk)

    # Create and store embeddings in FAISS
    texts = [chunk["chunk"] for chunk in all_chunks]
    vector_store = FAISS.from_texts(texts, embeddings)

    # Optionally, store augmented data elsewhere or return it
    return vector_store, all_chunks

# File uploader for PDF and DOCX files
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process documents if they haven't been processed yet
if st.button("Process Documents"):
    documents = process_documents(uploaded_files)

    if documents:
        # Create and store embeddings, and get augmented chunks
        vector_store, all_chunks = create_embeddings_and_store(documents)
        st.session_state.vector_store = vector_store
        st.session_state.augmented_chunks = all_chunks
        st.success("Documents processed and embeddings created.")
    else:
        st.warning("No documents to process.")

# Display the chat-like interface for the conversation
st.header("Chat with your Documents")
for entry in st.session_state.history:
    st.write(f"**User:** {entry['query']}")
    st.write(f"**Assistant:** {entry['response']}")

# Text input for the user's query
query = st.text_input("Please enter your query:", key="user_query")

async def get_relevant_chunks(sub_query, retriever, top_k=5):
    # Retrieve the top K most relevant chunks asynchronously
    retrieved_docs = await retriever.ainvoke(sub_query)
    return [doc.page_content for doc in retrieved_docs[:top_k]]

if st.button("Ask") and query:
    # Ensure vector store is available
    if st.session_state.vector_store:
        vector_store = st.session_state.vector_store
        retriever = VectorStoreRetriever(vectorstore=vector_store)

        sub_queries = query.split('?')
        responses = []

        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:
                top_chunks = asyncio.run(get_relevant_chunks(sub_query, retriever, top_k=5))

                if top_chunks:
                    augmented_data = [chunk for chunk in st.session_state.augmented_chunks if chunk["chunk"] in top_chunks]

                    user_prompt = sub_query + "\n\n" + "\n\n".join(
                        f"Chunk {i + 1}: {chunk['chunk'][:200]}... Title: {chunk['title']}, Keywords: {', '.join(chunk['keywords'])}"
                        for i, chunk in enumerate(augmented_data)
                    )

                    # Including system prompt with each sub-query
                    response = client.chat.completions.create(
                        model="jamba-1.5-mini",
                        messages=[
                            ChatMessage(role="system", content=system_prompt),
                            ChatMessage(role="user", content=user_prompt)
                        ],
                        stream=True
                    )

                    refined_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            refined_response += chunk.choices[0].delta.content

                    responses.append(f"**{sub_query}**: {refined_response.strip()}" if refined_response else f"**{sub_query}**: No relevant response.")

                else:
                    responses.append(f"**{sub_query}**: No relevant chunks retrieved.")

        final_response = "\n\n".join(responses)
        st.write("Refined Response:", final_response)

    else:
        st.warning("Please process documents before querying.")

