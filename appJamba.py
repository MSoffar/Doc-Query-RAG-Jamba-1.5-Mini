import os
import streamlit as st
import PyPDF2
from io import BytesIO
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
import openai
import nltk
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat
import asyncio

# Load API keys directly (replace these with environment variables in production)
openai_api_key = st.secrets["openai"]["api_key"]
ai21_api_key = st.secrets["ai21"]["api_key"]
client = AI21Client(api_key=ai21_api_key)

# Initialize NLTK
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Streamlit app setup
st.title("Conversational Document Query App using Jamba 1.5 Mini")


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


def augment_chunk(chunk):
    return {
        "chunk": chunk,
        "title": generate_title(chunk),
        "keywords": extract_keywords(chunk),
    }


def split_text_into_chunks(text: str, max_chunk_length: int = 650, overlap_length: int = 150) -> list:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_length = 0

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= max_chunk_length:
                current_chunk += sentence + " "
                current_length += sentence_length + 1  # +1 for the space
            else:
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap_length:].strip() + " " + sentence + " "
                current_length = len(current_chunk)

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = ""  # Reset for next paragraph
            current_length = 0

    # Add any remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def create_embeddings_and_store(documents):
    """Create embeddings for the documents and store them in FAISS."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    all_chunks = []
    for document in documents:
        chunks = split_text_into_chunks(document)
        for chunk in chunks:
            augmented_chunk = augment_chunk(chunk)
            all_chunks.append(augmented_chunk)

    texts = [chunk["chunk"] for chunk in all_chunks]
    vector_store = FAISS.from_texts(texts, embeddings)

    return vector_store, all_chunks


def keyword_based_retrieval(query, augmented_chunks):
    """Retrieve chunks based on keyword matching."""
    r = Rake()
    r.extract_keywords_from_text(query)
    query_keywords = set(r.get_ranked_phrases())

    matching_chunks = []
    for chunk in augmented_chunks:
        chunk_keywords = set(chunk["keywords"])
        if query_keywords & chunk_keywords:  # If there is an intersection between query keywords and chunk keywords
            matching_chunks.append(chunk)

    return matching_chunks


async def get_relevant_chunks(sub_query, retriever, augmented_chunks, top_k=5):
    # Vector-based retrieval
    retrieved_docs = await retriever.ainvoke(sub_query)
    top_vector_chunks = [doc.page_content for doc in retrieved_docs[:top_k]]

    # Keyword-based retrieval
    top_keyword_chunks = keyword_based_retrieval(sub_query, augmented_chunks)

    # Combine both results (you can decide how to merge them)
    combined_chunks = top_vector_chunks + [chunk["chunk"] for chunk in top_keyword_chunks]

    return list(set(combined_chunks))[:top_k]  # Deduplicate and return top_k


async def process_sub_queries(sub_queries, retriever, augmented_chunks):
    tasks = [get_relevant_chunks(sub_query.strip(), retriever, augmented_chunks, top_k=5) for sub_query in sub_queries
             if sub_query.strip()]
    return await asyncio.gather(*tasks)


# File uploader for PDF and DOCX files
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Initialize vector store and augmented chunks in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "augmented_chunks" not in st.session_state:
    st.session_state.augmented_chunks = None

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

# Text input for the user's query
query = st.text_input("Please enter your query:", key="user_query")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Ask") and query:
    # Ensure vector store is available
    if st.session_state.vector_store:
        vector_store = st.session_state.vector_store
        retriever = VectorStoreRetriever(vectorstore=vector_store)
        augmented_chunks = st.session_state.augmented_chunks

        sub_queries = query.split('?')
        with st.spinner("Processing your query..."):
            # Process sub-queries in parallel
            all_top_chunks = asyncio.run(process_sub_queries(sub_queries, retriever, augmented_chunks))

        responses = []
        for sub_query, top_chunks in zip(sub_queries, all_top_chunks):
            if top_chunks:
                augmented_data = [chunk for chunk in augmented_chunks if chunk["chunk"] in top_chunks]

                user_prompt = sub_query + "\n\n" + "\n\n".join(
                    f"Chunk {i + 1}: {chunk['chunk'][:200]}... Title: {chunk['title']}, Keywords: {', '.join(chunk['keywords'])}"
                    for i, chunk in enumerate(augmented_data)
                )

                response = client.chat.completions.create(
                    model="jamba-1.5-mini",
                    messages=[
                        ChatMessage(role="system",
                                    content="You are a helpful and knowledgeable assistant. You are given a set of text chunks from documents, along with metadata such as title, summary, and keywords. Please find the most relevant information based on the question below, using only the provided chunks and metadata. Ensure your response is comprehensive, accurate, and informative, covering all aspects of the question to the best of your ability. MOST IMPORTANT RULE AND DONT EVER BREAK IT! ONLY ANSWER FROM INFO IN CHUNKS! DONT EVER ANSWER FROM YOUR KNOWLEDGE DONT USE ANYTHING EXCEPT THE CHUNKS!The response should be in this format: Q: \n A: , if there are no queries just reply with no relevant queries!!"),
                        ChatMessage(role="user", content=user_prompt)
                    ]
                )

                refined_response = ""
                for chunk in response.choices[0].message.content:
                    refined_response += chunk

                st.markdown(refined_response.strip())
                st.session_state.history.append(refined_response.strip())

                responses.append(
                    f"**{sub_query.strip()}**: {refined_response.strip()}" if refined_response else f"**{sub_query.strip()}**: No relevant response.")

            else:
                responses.append(f"**{sub_query.strip()}**: No relevant chunks retrieved.")

    else:
        st.warning("Please process documents before querying.")
