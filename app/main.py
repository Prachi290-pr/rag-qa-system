from fastapi import FastAPI
from app.api.upload import router as upload_router

app = FastAPI(title="RAG-Based QA System")

app.include_router(upload_router)


@app.get("/")
def root():
    return {"message": "RAG API is running"}