from app.services.rag_services import (
    ingest_document,
    retrieve_chunks,
    save_index_and_chunks,
    load_index_and_chunks
)

PDF_PATH = "data/uploaded_pdfs/sample.pdf"

print("Ingesting Document...")
index, chunks = ingest_document(PDF_PATH)

print(f"Total chunks created: {len(chunks)}")

print("\n Testing retrieval..")
query = "What is RAG and why is it used?"

results = retrieve_chunks(query, index, chunks, top_k=2)

print("\n Retrieved Chunks:\n")

for i, chunk in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(chunk[:300])
    print()




print("💾 Saving index and chunks...")
save_index_and_chunks(index, chunks)

print("📂 Loading index and chunks...")
index, chunks = load_index_and_chunks()

print(f"✅ Total chunks loaded: {len(chunks)}")