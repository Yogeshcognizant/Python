import streamlit as st
import os
from datetime import datetime
from complete_rag_system import *

# Page config
st.set_page_config(
    page_title="Cognizant AI Support Chatbot", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #2e86ab;
    background-color: #f8f9fa;
}
.user-message {
    background-color: #e3f2fd;
    border-left-color: #1976d2;
}
.bot-message {
    background-color: #f1f8e9;
    border-left-color: #388e3c;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached to avoid reloading)"""
    try:
        embedding_manager = EmbeddingManager()
        vectorstore = VectorStore()
        
        if vectorstore.collection.count() > 0:
            rag_retriever = RAGRetriever(vectorstore, embedding_manager)
            groq_llm = GroqLLM()
            return rag_retriever, groq_llm.llm, vectorstore.collection.count()
        else:
            with st.spinner("🔧 Initializing AI Knowledge Base..."):
                retriever, llm, _ = main()
                return retriever, llm, retriever.vector_store.collection.count()
    except Exception as e:
        st.error("⚠️ System temporarily unavailable. Please contact support.")
        return None, None, 0

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Cognizant AI Support Chatbot</h1>
    <p>Intelligent Document Assistant | Powered by Advanced RAG Technology</p>
</div>
""", unsafe_allow_html=True)

# Initialize system
if st.session_state.rag_system is None:
    retriever, llm, doc_count = initialize_rag_system()
    if retriever and llm:
        st.session_state.rag_system = (retriever, llm, doc_count)
        st.success("✅ AI Assistant Ready!")
    else:
        st.error("❌ Service unavailable. Please try again later.")
        st.stop()

# Main interface
if st.session_state.rag_system:
    retriever, llm, doc_count = st.session_state.rag_system
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>📚</h3><p>Knowledge Base</p><h4>{} Documents</h4></div>'.format(doc_count), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>💬</h3><p>Conversations</p><h4>{} Queries</h4></div>'.format(len(st.session_state.chat_history)), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🤖</h3><p>AI Assistant</p><h4>Active</h4></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>⚡</h3><p>Status</p><h4>Online</h4></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chat interface
    st.subheader("💬 Ask Your Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Question", placeholder="Type your question here... (e.g., 'What is machine learning?')", label_visibility="hidden")
    with col2:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    # Process question
    if (ask_button and question) or (question and False):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                answer = rag_simple(question, retriever, llm)
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    "question": question, 
                    "answer": answer, 
                    "timestamp": timestamp
                })
                st.rerun()
            except Exception as e:
                st.error("⚠️ Unable to process your request. Please try again.")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📋 Conversation History")
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You ({chat['timestamp']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 AI Assistant:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 **Getting Started:** Ask me anything about the documents in our knowledge base!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏢 <strong>Cognizant Technology Solutions</strong> | Powered by Advanced AI & Machine Learning</p>
    <p>📧 For technical support, contact: <strong>ai-support@cognizant.com</strong></p>
</div>
""", unsafe_allow_html=True)