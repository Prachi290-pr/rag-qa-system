import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

# Initialize models
# Embedding model for converting text to vectors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Generative model for answering questions
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")  


def load_pdf(file_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise ValueError("No extractable text found in PDF. Try another PDF.")

    return text


def chunk_text(text, chunk_size=800, overlap=100):
    """Splits text into chunks with overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):
    """Generates vector embeddings for a list of text chunks."""
    return embedding_model.encode(chunks)


def store_in_faiss(embeddings):
    """Stores embeddings in a FAISS index for similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def ingest_document(file_path):
    """Full pipeline: Load PDF -> Chunk -> Embed -> Store in FAISS."""
    # 1. Load text
    text = load_pdf(file_path)

    # 2. Chunk text
    chunks = chunk_text(text)

    # 3. Create embeddings
    embeddings = create_embeddings(chunks)

    # 4. Create FAISS index
    index = store_in_faiss(embeddings)

    return index, chunks


def retrieve_chunks(query, index, chunks, top_k=2):
    """Searches the FAISS index for the most relevant chunks."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def save_index_and_chunks(index, chunks, save_path="data/index_store"):
    """Saves the FAISS index and chunks to disk."""
    os.makedirs(save_path, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, f"{save_path}/faiss.index")

    # Save chunks
    with open(f"{save_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index_and_chunks(save_path="data/index_store"):
    """Loads the FAISS index and chunks from disk."""
    if not os.path.exists(f"{save_path}/faiss.index"):
        return None, None

    index = faiss.read_index(f"{save_path}/faiss.index")

    with open(f"{save_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def generate_rag_answer(query, index, chunks, top_k=3):
    """Retrieves context and uses Gemini to answer the user's question."""
    
    # 1. Retrieve relevant context
    retrieved_chunks = retrieve_chunks(query, index, chunks, top_k=top_k)
    context = "\n\n".join(retrieved_chunks)

    # 2. Construct the prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the context provided.
    If the answer is not present in context, say:
    "I could not find this information in the document."

    --------------------
    Context:
    {context}
    --------------------

    Question:
    {query}

    Answer:
    """

    # 3. Generate response
    response = gemini_model.generate_content(prompt)
    return response.text