from pypdf import PdfReader
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np 
import pickle
import os


# Loading embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


# 1. Load PDF

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text=""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: 
            text += page_text + "\n"

    if text.strip() == "":
        raise ValueError("No extractable text found in PDF. Try another PDF.")

    return text

# 2. Chunk Text

def chunk_text(text, chunk_size=800, overlap=100):
    chunks=[]
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 3. Create Embeddings

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings 


# 4. Store in FAISS

def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index



# 5. FULL Ingestion Pipeline

def ingest_document(file_path):

    # Load text
    text = load_pdf(file_path)

    # Chunk
    chunks = chunk_text(text)

    # Embeddings 
    embeddings = create_embeddings(chunks)

    # Store in FAISS
    index = store_in_faiss(embeddings)

    return index, chunks


# 6. Retrieval 

def retrieve_chunks(query, index, chunks, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]



# 7. Save FAISS + Chunks 

def save_index_and_chunks(index, chunks, save_path="data/index_store"):

    os.makedirs(save_path, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, f"{save_path}/faiss.index")

    # Save chunks
    with open(f"{save_path}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


# 8. Load FAISS

def load_index_and_chunks(save_path="data/index_store"):

    index = faiss.read_index(f"{save_path}/faiss.index")

    with open(f"{save_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks