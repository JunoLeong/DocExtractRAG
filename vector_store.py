import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embeddings Initialized")

# Clean the content (remove URLs, extra whitespaces, metadata, etc.)
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Optionally, remove metadata or any irrelevant content (e.g., page numbers, author details)
    text = re.sub(r'^[a-zA-Z\s]*\d{1,2}\s*\n?', '', text)  # Remove page info (if exists)
    return text.strip()

loader = PyPDFLoader("Prompt_Engineering_in_LLM.pdf")
documents = loader.load()

print("Documents Loaded...")

# Clean the document text
cleaned_documents = [clean_text(doc.page_content) for doc in documents]

# Split text into chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=45)
texts = text_splitter.split_documents(documents)

#print(texts[0])

vector_store = Chroma.from_documents(texts, 
                                     embeddings, 
                                     collection_metadata={"hnsw:space": "cosine"}, 
                                     persist_directory="stores/doc_cosine")

print("Vector Store is Created")