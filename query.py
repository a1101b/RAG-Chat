import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LOAD ENVIRONMENT VARIABLES
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CONNECT TO EMBEDDING MODEL
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# CONNECT TO EXISTING PINECONE INDEX
vector_store = PineconeVectorStore(
    index_name="rag-project",
    embedding=embeddings
)

# SET UP RETRIEVER FOR 3 CHUNKS
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# SET UP GROQ LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# BUILD THE PROMPT TEMPLATE TO SEND TO LLM
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

# HELPER FUNCTION TO FORMAT RETRIEVED CHUNKS
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# BUILD THE CHAIN USING LCEL 
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# QUESTION LOOPING
print("\nRAG system ready. Type 'quit' to exit.\n")

while True:
    question = input("Ask a question: ")
    
    if question.lower() == "quit":
        print("Goodbye!")
        break
    
    if question.strip() == "":
        print("Please enter a question.")
        continue
    
    answer = chain.invoke(question)
    print("\n--- ANSWER ---")
    print(answer)
    print()