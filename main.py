import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Load the Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to embed text chunks
def embed_text_chunks(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings

# Function to create a FAISS vector store
def create_vectorstore(embeddings, text_chunks):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_chunks

# Function to query the FAISS vector store
def query_vectorstore(query, index, text_chunks, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)
    return [text_chunks[idx] for idx in indices[0]]

# Function to call the Gemini API with RAG context
def call_gemini_api(context, question):
    import google.generativeai as genai

    # Configure the Gemini API with your API key
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Construct the prompt
    prompt = f"""You are an intelligent assistant. Use the following context to answer the question:
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    try:
        # Load the Gemini model and generate a response
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        return response.text if hasattr(response, 'text') else "No text response from the model."
    except Exception as e:
        return f"Error occurred: {str(e)}"



# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.header("CHAT WITH PDF :books:")

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    # Step 1: Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2: Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Step 3: Embed the text chunks
                    embeddings = embed_text_chunks(text_chunks)

                    # Step 4: Create a FAISS vector store
                    index, chunk_store = create_vectorstore(embeddings, text_chunks)

                    st.success("Documents processed and vector store created!")
                    st.session_state['index'] = index
                    st.session_state['chunk_store'] = chunk_store
            else:
                st.warning("Please upload at least one PDF.")

    # User query input
    query = st.text_input("Ask a question based on your uploaded documents:")

    if query and 'index' in st.session_state and 'chunk_store' in st.session_state:
        with st.spinner("Searching for the best answers..."):
            # Step 5: Query the FAISS vector store
            model = SentenceTransformer('all-MiniLM-L6-v2')
            index = st.session_state['index']
            chunk_store = st.session_state['chunk_store']

            results = query_vectorstore(query, index, chunk_store, model)

            # Combine the retrieved chunks into a single context
            context = " ".join(results)

            # Step 6: Call the Gemini API for a refined response
            llm_response = call_gemini_api(context, query)
            st.subheader("Answer:")
            st.write(llm_response)

if __name__ == "__main__":
    main()
