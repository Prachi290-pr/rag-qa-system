from fastapi import APIRouter, UploadFile, File
import os
import shutil

from app.services.rag_services import ingest_document, save_index_and_chunks

router = APIRouter()


UPLOAD_DIR = "data/uploaded_pdfs"


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Ensure folder exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run ingestion
    index, chunks = ingest_document(file_path)

    # Save index + chunks
    save_index_and_chunks(index, chunks)

    return {
        "message": "Document uploaded and processed",
        "filename": file.filename,
        "chunks_created": len(chunks)
    }