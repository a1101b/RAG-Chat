import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# LOAD ENVIRONMENT VARIABLES
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# LOAD  DOCUMENT
loader = PyPDFLoader("documents/example.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# SPLIT INTO CHUNKS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = splitter.split_documents(pages)
print(f"Split into {len(docs)} chunks")

# SET UP EMBEDDING MODEL
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# CONNECT TO PINECONE WITH API KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "rag-project"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
print(f"Connected to Pinecone index: {INDEX_NAME}")

# STORE CHUNKS IN PINECONE
vector_store = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=INDEX_NAME
)
print("Done — all chunks stored in Pinecone")