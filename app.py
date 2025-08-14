import os
import fitz  # PyMuPDF
import faiss
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")  # Make sure your .env has HF_API_KEY
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")

if HF_API_KEY is None:
    st.error("❌ Hugging Face API key not found. Add HF_API_KEY to your .env file.")
    st.stop()

# ------------------------------
# Initialize embedding model
# ------------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# PDF Text Extraction
# ------------------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# ------------------------------
# Chunking function
# ------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# ------------------------------
# Hugging Face API call
# ------------------------------
def query_huggingface(prompt):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300, "temperature": 0.5, "do_sample": False}
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL_ID}",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()[0]["generated_text"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="StudyMate", layout="wide")
st.title("📚 StudyMate – AI PDF Study Assistant")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    sources = []
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        sources.extend([uploaded_file.name] * len(chunks))

    # ------------------------------
    # Create embeddings & FAISS index
    # ------------------------------
    embeddings = embedder.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("✅ Documents processed! Ask your question below.")

    query = st.text_input("Ask a question:")
    if query:
        q_embed = embedder.encode([query]).astype("float32")
        distances, indices = index.search(q_embed, k=3)

        retrieved_chunks = [all_chunks[i] for i in indices[0]]
        context = "\n".join(retrieved_chunks)

        prompt = f"Answer based strictly on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"
        try:
            answer = query_huggingface(prompt)
            st.markdown(f"### 💡 Answer:\n{answer}")
        except Exception as e:
            st.error(f"❌ Error querying Hugging Face API: {e}")

        with st.expander("Referenced Paragraphs"):
            for chunk in retrieved_chunks:
                st.write(chunk)