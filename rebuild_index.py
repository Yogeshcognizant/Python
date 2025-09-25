from complete_rag_system import rebuild_index

if __name__ == "__main__":
    print("🔄 Rebuilding RAG index with new PDFs...")
    success = rebuild_index()
    if success:
        print("✅ Index rebuilt successfully! You can now run your Streamlit app.")
    else:
        print("❌ Failed to rebuild index.")