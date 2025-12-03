import os
import sys
import argparse
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import config

def main():
    # 1. Initialize Embeddings
    print("Initializing Embeddings...")
    embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_BASE_URL,
        model=config.MODEL_EMBEDDING
    )

    # 2. Load Vector DB
    print(f"Loading Vector DB from {config.CHROMA_PERSIST_DIRECTORY}...")
    if not os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        print(f"❌ Error: Vector DB directory '{config.CHROMA_PERSIST_DIRECTORY}' not found.")
        print("Please run 'main.py' first to collect papers and create the database.")
        sys.exit(1)

    vector_db = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME
    )

    # 3. Initialize LLM
    print(f"Initializing LLM ({config.MODEL_GENERATOR})...")
    llm = ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.MODEL_GENERATOR,
        temperature=config.GENERATOR_TEMPERATURE
    )

    # 4. Define Prompt Template
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Define RAG Chain
    retriever = vector_db.as_retriever(search_kwargs={"k": config.VECTOR_DB_SEARCH_K})
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Chat Loop
    print("\n🤖 RAG Chatbot is ready! (Type 'exit', 'quit', or 'q' to stop)")
    print("-" * 50)

    while True:
        try:
            query = input("\nUser: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue

            print("Thinking...", end="", flush=True)
            
            # Stream the response
            print("\rBot: ", end="", flush=True)
            full_response = ""
            for chunk in chain.stream(query):
                print(chunk, end="", flush=True)
                full_response += chunk
            print() # Newline after response

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
