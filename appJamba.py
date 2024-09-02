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

def process_documents(uploaded_files, urls):
    """Process PDF and DOCX files and URLs."""
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
        "keywords": extract_keywords(chunk),
        #"summary": generate_summary(chunk),
        # "entities": extract_entities(chunk),
        # "questions": generate_questions(chunk),
        # "source": "Document X, Page Y"  # Replace with actual source info if available
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
        # Only process if there are new documents uploaded
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

        # Split the query into sub-queries if needed
        sub_queries = query.split('?')

        responses = []

        # Process each sub-query
        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:
                # Run the async task to get the relevant chunks for each sub-query
                top_chunks = asyncio.run(get_relevant_chunks(sub_query, retriever, top_k=5))

                if top_chunks:
                    # Retrieve augmented data from session state
                    augmented_data = [chunk for chunk in st.session_state.augmented_chunks if chunk["chunk"] in top_chunks]

                    # Create a prompt using the retrieved chunks and metadata
                    system_prompt = (
                        "You are a highly accurate and detail-oriented assistant. You are provided with specific text chunks extracted from documents, including relevant metadata such as titles, summaries, and keywords. "
                        "Your task is to generate responses strictly based on the information within these chunks. Under no circumstances should you utilize external knowledge or provide information not contained within the provided chunks. "
                        "When answering a query, ensure that your response is clear, concise, and directly relevant to the question asked. "
                        "If the query does not match any relevant information in the chunks, respond with 'No relevant information available in the provided chunks.' "
                        "It is critical that your answers are derived solely from the content within the chunks provided. Do not infer, assume, or use any information outside of the provided chunks."
                    )

                    user_prompt = sub_query + "\n\n" + "\n\n".join(
                        f"Chunk {i + 1}: {chunk['chunk'][:200]}... Title: {chunk['title']}, Keywords: {', '.join(chunk['keywords'])}"
                        for i, chunk in enumerate(augmented_data)
                    )

                    response = client.chat.completions.create(
                        model="jamba-1.5-mini",
                        messages=[
                            ChatMessage(role="system", content=system_prompt),
                            ChatMessage(role="user", content=user_prompt)
                        ]
                    )

                    refined_response = response['choices'][0]['content'].strip()
                    responses.append(f"**{sub_query}**: {refined_response}")

                else:
                    responses.append(f"**{sub_query}**: No relevant chunks retrieved.")

        # Join and display all responses
        final_response = "\n\n".join(responses)
        st.write("Refined Response:", final_response)

        # Save conversation history
        st.session_state.history.append({"query": query, "response": final_response})

        # Clear the input box after processing
        query = ""
    else:
        st.warning("Please process documents before querying.")

