from fastapi import APIRouter
from pydantic import BaseModel

from app.services.rag_services import (
    load_index_and_chunks,
    generate_rag_answer
)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
async def query_rag(request: QueryRequest):

    # Load saved index + chunks
    index, chunks = load_index_and_chunks()

    # Generate answer
    answer = generate_rag_answer(request.query, index, chunks)

    return {
        "query": request.query,
        "answer": answer
    }