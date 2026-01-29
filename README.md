# Retrieval-Augmented Generation (RAG) Based Question Answering System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that combines document retrieval with Large Language Model (LLM) generation to produce context-aware and factually grounded answers from a custom knowledge base.

The system improves response accuracy by retrieving relevant document chunks and injecting them into the LLM prompt before response generation, helping reduce hallucinated outputs.

---

## Objectives
- Build an end-to-end RAG pipeline for document-based question answering  
- Improve factual accuracy using retrieval-based context grounding  
- Demonstrate practical LLM and vector database workflow  

---

## Tech Stack
- Python  
- LLM API (OpenAI / Gemini or similar)  
- Embeddings Model  
- Vector Database (FAISS / ChromaDB)  
- Pandas, NumPy  

---

## System Workflow
1. Load and preprocess documents  
2. Split documents into semantic chunks  
3. Generate embeddings for each chunk  
4. Store embeddings in vector database  
5. Convert user query into embedding  
6. Retrieve top relevant document chunks  
7. Inject context into LLM prompt  
8. Generate final grounded response  



