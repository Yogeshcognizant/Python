from complete_rag_system import *

# Skip PDF processing if vector store exists
embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
rag_retriever = RAGRetriever(vectorstore, embedding_manager)

try:
    groq_llm = GroqLLM()
    llm = groq_llm.llm
    
    # Interactive queries
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = rag_simple(question, rag_retriever, llm)
        print(f"\nAnswer: {answer}")
        
except ValueError as e:
    print(f"Error: {e}")
    print("Please set your GROQ_API_KEY in the .env file")