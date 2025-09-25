import os
import numpy as np
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# ML/Vector imports
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

class PDFProcessor:
    """Handles PDF loading and document processing"""
    
    @staticmethod
    def process_all_pdfs(pdf_directory):
        """Process all PDF files in a directory"""
        all_documents = []
        pdf_dir = Path(pdf_directory)
        
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                
                all_documents.extend(documents)
                print(f"  ✓ Loaded {len(documents)} pages")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents

    @staticmethod
    def split_documents(documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        
        if split_docs:
            print(f"\nExample chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")
        
        return split_docs

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def rebuild_index(self):
        """Clear existing collection and rebuild from scratch"""
        print("Rebuilding vector store index...")
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print("Vector store cleared and ready for new documents")
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            raise

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

class GroqLLM:
    """Groq LLM wrapper for response generation"""
    
    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context"""
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )
        
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class AdvancedRAGPipeline:
    """Advanced RAG pipeline with multiple features"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(self, question: str, top_k: int = 8, min_score: float = 0.1, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        """Enhanced advanced query with comprehensive retrieval"""
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            
            prompt = f"""You are a comprehensive AI assistant. Provide a detailed and complete answer using all relevant information from the context. Include specific procedures, indicators, implementation details, and any other relevant information mentioned.

Context:
{context}

Question: {question}

Answer: Provide a thorough response covering all aspects mentioned in the context."""
            
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            
            response = self.llm.invoke([prompt])
            answer = response.content

        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

def rag_simple(query, retriever, llm, top_k=7):
    """Enhanced RAG function: retrieve more context + generate comprehensive response"""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=0.1)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found to answer the question."
    
    prompt = f"""You are a comprehensive AI assistant. Use the following context to provide a detailed and complete answer to the question. Include all relevant information, procedures, indicators, and implementation details mentioned in the context.

    Context:
    {context}

    Question: {query}

    Answer: Provide a thorough response that covers all aspects mentioned in the context. Include specific details, procedures, indicators, and any implementation information available."""
    
    response = llm.invoke([prompt])
    return response.content

def rag_advanced(query, retriever, llm, top_k=8, min_score=0.1, return_context=False):
    """Enhanced RAG pipeline with comprehensive retrieval"""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])
    
    prompt = f"""You are a comprehensive AI assistant. Provide a detailed and complete answer using all relevant information from the context. Include specific procedures, indicators, implementation details, and any other relevant information.

Context:
{context}

Question: {query}

Answer: Provide a thorough response covering all aspects mentioned in the context."""
    response = llm.invoke([prompt])
    
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output

def rebuild_index():
    """Rebuild the entire RAG index with new PDFs"""
    print("🔄 REBUILDING RAG INDEX...")
    
    # 1. Process PDFs
    print("\n1. Processing PDFs...")
    all_pdf_documents = PDFProcessor.process_all_pdfs("data/pdf_files")
    
    # 2. Split documents
    print("\n2. Splitting documents...")
    chunks = PDFProcessor.split_documents(all_pdf_documents)
    
    # 3. Initialize embedding manager
    print("\n3. Initializing embedding manager...")
    embedding_manager = EmbeddingManager()
    
    # 4. Initialize vector store and rebuild
    print("\n4. Rebuilding vector store...")
    vectorstore = VectorStore()
    vectorstore.rebuild_index()
    
    # 5. Generate embeddings and store
    print("\n5. Generating embeddings...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vectorstore.add_documents(chunks, embeddings)
    
    print("\n✅ INDEX REBUILD COMPLETE!")
    print(f"Total documents: {len(all_pdf_documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector store count: {vectorstore.collection.count()}")
    
    return True

def main():
    """Main function to run the complete RAG pipeline"""
    print("Starting RAG System Setup...")
    
    # 1. Process PDFs
    print("\n1. Processing PDFs...")
    all_pdf_documents = PDFProcessor.process_all_pdfs("data/pdf_files")
    
    if not all_pdf_documents:
        print("❌ No documents found! Please add PDF files to process.")
        return None, None, None
    
    # 2. Split documents
    print("\n2. Splitting documents...")
    chunks = PDFProcessor.split_documents(all_pdf_documents)
    
    if not chunks:
        print("❌ No document chunks created!")
        return None, None, None
    
    # 3. Initialize embedding manager
    print("\n3. Initializing embedding manager...")
    embedding_manager = EmbeddingManager()
    
    # 4. Initialize vector store
    print("\n4. Initializing vector store...")
    vectorstore = VectorStore()
    
    # 5. Generate embeddings and store
    print("\n5. Generating embeddings...")
    texts = [doc.page_content for doc in chunks]
    if not texts:
        print("❌ No text content found in documents!")
        return None, None, None
    
    embeddings = embedding_manager.generate_embeddings(texts)
    if len(embeddings) == 0:
        print("❌ Failed to generate embeddings!")
        return None, None, None
    
    vectorstore.add_documents(chunks, embeddings)
    
    # 6. Initialize retriever
    print("\n6. Initializing retriever...")
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    
    # 7. Initialize LLM
    print("\n7. Initializing LLM...")
    try:
        groq_llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))
        llm = groq_llm.llm
        print("Groq LLM initialized successfully!")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Please set your GROQ_API_KEY environment variable to use the LLM.")
        return None, None, None
    
    # 8. Initialize advanced RAG pipeline
    print("\n8. Initializing advanced RAG pipeline...")
    adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
    
    print("\n✅ RAG System Setup Complete!")
    print(f"Total documents: {len(all_pdf_documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector store count: {vectorstore.collection.count()}")
    
    return rag_retriever, llm, adv_rag

if __name__ == "__main__":
    retriever, llm, advanced_rag = main()
    
    if retriever and llm:
        # Example queries
        print("\n" + "="*50)
        print("EXAMPLE QUERIES")
        print("="*50)
        
        # Simple RAG example
        print("\n1. Simple RAG Query:")
        answer = rag_simple("What is debugging in python?", retriever, llm)
        print(answer)
        
        # Advanced RAG example
        print("\n2. Advanced RAG Query:")
        result = rag_advanced("what is exception in python", retriever, llm, top_k=3, min_score=0.1, return_context=True)
        print("Answer:", result['answer'])
        print("Confidence:", result['confidence'])
        
        # Advanced pipeline example
        print("\n3. Advanced Pipeline Query:")
        result = advanced_rag.query("what is try catch block in python", top_k=3, min_score=0.1, summarize=True)
        print("Answer:", result['answer'])
        print("Summary:", result['summary'])