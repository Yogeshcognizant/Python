from complete_rag_system import *

# Quick setup and query
retriever, llm, advanced_rag = main()

if retriever and llm:
    # Ask questions
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = rag_simple(question, retriever, llm)
        print(f"\nAnswer: {answer}")