import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

print("1. Reconnecting to ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("2. Booting up the LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

print("3. Building the memory-aware pipeline...")
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# --- NEW: The Contextualizer ---
# This prompt tells the AI to look at the chat history and the new question, 
# and rewrite the question so it makes sense without the history.
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# This creates a new retriever that rewrites the question first
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# --- NEW: The Answering Prompt ---
# We update the prompt to accept the chat_history
qa_system_prompt = """You are a technical assistant. Answer the question using ONLY the provided context from the git documentation.
If you don't know the answer based on the context, just say that you don't know.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- NEW: The Interactive Loop ---
print("\n=== GitDocs RAG Engine Initialized ===")
print("Type 'exit' or 'quit' to stop.")

chat_history = [] # This list will hold the "consciousness"

while True:
    user_query = input("\nYou: ")
    if user_query.lower() in ['exit', 'quit']:
        break
        
    response = rag_chain.invoke({
        "input": user_query,
        "chat_history": chat_history
    })
    
    print(f"\nAI: {response['answer']}")
    
    # Save this turn to the memory array so the AI remembers it for the next loop
    chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=response["answer"])
    ])