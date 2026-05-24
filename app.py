import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec

# LOAD ENVIRONMENT VARIABLES
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "rag-project"

# SET UP FASTAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("static/index.html")

# SET UP SHARED COMPONENTS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# MODELS FOR ENDPOINTS
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# ENDPOINTS
@app.get("/")
def root():
    return {"status": "RAG API is running"}

# ASK QUESTION ENDPOINT
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = chain.invoke(request.question)
    return AnswerResponse(question=request.question, answer=answer)

# UPLOAD ENDPOINT TO ADD NEW DOCUMENTS
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # ONLY ALLOW PDFS
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # SAVE UPLOADED FILE TO DOCUMENTS FOLDER
    os.makedirs("documents", exist_ok=True)
    file_path = f"documents/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # LOAD AND SPLIT DOCUMENT
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.split_documents(pages)

    # ADD METADATA TO KNOW WHICH FILE EACH CHUNK CAME FROM
    for doc in docs:
        doc.metadata["source_file"] = file.filename

    # STORE CHUNKS IN PINECONE
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    return {
        "message": f"Successfully indexed {file.filename}",
        "pages": len(pages),
        "chunks": len(docs)
    }