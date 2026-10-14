import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the GOOGLE_API_KEY from .env
load_dotenv()

print("1. Reconnecting to ChromaDB...")
# CRITICAL: We must use the exact same embedding model used to build the database
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("2. Booting up the LLM...")
# We use Gemini 1.5 Flash to act as the "brain" that reads the retrieved chunks
# temperature=0 ensures it gives factual answers instead of trying to be "creative"
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

print("3. Building the pipeline...")
# Configure the database to return the 2 most relevant text chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Set strict rules for the AI
prompt = ChatPromptTemplate.from_template("""
You are a technical assistant. Answer the question using ONLY the provided context from the git documentation.
If you don't know the answer based on the context, just say that you don't know.

Context: {context}

Question: {input}
""")

# Link everything together into a "chain"
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 4. Ask the question!
user_query = "summarize this repo in 2-3 lines"
print(f"\nQuestion: {user_query}")
print("Thinking...")

response = rag_chain.invoke({"input": user_query})

print("\n--- FINAL ANSWER ---")
print(response["answer"])

print("\n--- SOURCES USED ---")
for doc in response["context"]:
    print(f"- {doc.metadata['source']}")