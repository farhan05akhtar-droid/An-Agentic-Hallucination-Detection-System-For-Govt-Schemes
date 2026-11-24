import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router
from rag.vector_store import get_retriever

app = FastAPI(
    title="GovScheme AI Agent API",
    description="Agentic Hallucination Detection System in Government Welfare Schemes",
    version="1.0.0"
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Warm up embedding model and ChromaDB connection on startup."""
    print("Pre-warming vector retriever and embedding model...")
    try:
        retriever = get_retriever(k=1)
        _ = retriever.invoke("startup warm up")
        print("Vector store retriever pre-warmed successfully!")
    except Exception as e:
        print(f"Startup warm-up notice: {e}")

@app.get("/")
def root():
    return {
        "message": "GovScheme AI Multi-Agent API is running.",
        "documentation": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
