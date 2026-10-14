import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load the GOOGLE_API_KEY from .env
load_dotenv()

DOCS_DIR = "./git_docs" 

print("1. Loading markdown files...")
loader = DirectoryLoader(DOCS_DIR, glob="**/*.md")
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

print("2. Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

print("3. Generating embeddings with Gemini and building ChromaDB...")
# We swap OpenAI for Gemini's latest text embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("Success! Database built with Gemini.")