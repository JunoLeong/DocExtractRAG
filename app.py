from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from ctransformers import AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import Runnable
from langchain.prompts.base import StringPromptValue

import os

# libraries for UI
import gradio as gr

# local_llm = "zephyr-7b-beta.Q4_K_M.gguf"

class CTransformersRunnable(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input, config=None, **kwargs):
        # Convert StringPromptValue to plain string
        if isinstance(input, StringPromptValue):
            input = input.to_string()
        
        # Optionally, handle the 'stop' sequences
        stop = kwargs.get('stop', None)
        if stop:
            print(f"Stop sequences: {stop}")
        
        # Pass processed input to the LLM
        return self.llm(input)

    async def _acall(self, input, config=None, **kwargs):
        # Handle async calls (same logic)
        if isinstance(input, StringPromptValue):
            input = input.to_string()
        
        stop = kwargs.get('stop', None)
        if stop:
            print(f"Stop sequences (async): {stop}")

        return self.llm(input)
      
config = {
'max_new_tokens': 1024,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'device': 'cpu',
'threads': int(os.cpu_count() / 2),
}

# llm = CTransformers(
#     model="./models/zephyr-7B-beta-GGUF",
#     model_type="zephyr",
#     config=config
# )

llm = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GGUF", 
                                           model_file="zephyr-7b-beta.Q4_K_M.gguf", 
                                           model_type="mistral",
                                           gpu_layers=50,
                                           max_new_tokens = 1000,
                                           context_length = 6000)
print("LLM Initialized...")

# Wrap the LLM in the custom Runnable
llm_runnable = CTransformersRunnable(llm)

custom_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=custom_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/doc_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})

# query = "What is Chain-of-Thought (CoT) Prompting?"
# semantic_search = retriever.invoke(query)
# print(semantic_search)

# print("--------------------------------------------------")

# chain_type_kwargs = {"prompt": prompt}

# qa = RetrievalQA.from_chain_type(
#     llm=llm_runnable,  # Use the wrapped LLM
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt},  # Pass the updated prompt
#     verbose=True
# )

# response = qa.invoke(query)

# print(response)

def get_response(input):
  query = input
  qa = RetrievalQA.from_chain_type(llm=llm_runnable, 
                                   chain_type="stuff", 
                                   retriever=retriever, 
                                   return_source_documents=True, 
                                   chain_type_kwargs={"prompt": prompt}, 
                                   verbose=True)
  response = qa.invoke(query)
  #return response
  
  # Extract the result and source documents
  result = response.get('result', "No result available")
  source_docs = response.get('source_documents', [])
    
    # Prepare a readable format
  sources = "\n".join(
        f"Source {i + 1}: {doc.metadata.get('source', 'Unknown source')} (Page {doc.metadata.get('page', 0)+1 })"
        for i, doc in enumerate(source_docs)
    )

  organized_response = f"""
{result}

### Sources:
{sources}
    """
  return organized_response
  

#Gradio as simple UI for user to react with the model
sample_prompts = ["What is prompt engineering?", 
                  "How to do new tasks without extensive training by using prompt engineering?", 
                  "What are the ways to reduce hallucination?"]

# Define Gradio interface
with gr.Blocks(theme=gr.themes.Base(), title="DocExtractRAG") as iface:
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: center;">
            <h1>Welcome to DocExtractRAG!</h1>
            <p>Enter your question below, and the system will retrieve relevant information from the document.</p>
            <p>The model is powered by the RAG system using Zephyr 7B Beta LLM.</p>
            </div>
            """
        )
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Ask Your Question:")
            input_box = gr.Textbox(
                label="Enter your prompt here:",
                placeholder="E.g., What is the title of this document?",
                lines=2,
                max_lines=3,
                interactive=True,
                container=False,
            )
            gr.Markdown("###### Example Questions:")
            example_buttons = gr.Examples(
                examples=sample_prompts,
                inputs=input_box,
                label="Click to insert an example question to ask"
            )
        
        # Output section
        with gr.Column(scale=2):
            gr.Markdown("### DocExtractRAG Response:")
            output_box = gr.Textbox(placeholder="The model response will appear here.",
                                    label=None, 
                                    interactive=False)
    
    with gr.Row():
      # Clear button
        clear_btn = gr.Button("Clear", variant="secondary")
        
        #Submit button
        submit_btn = gr.Button("Submit", variant="primary", size="lg")
        submit_btn.click(get_response, inputs=input_box, outputs=output_box)
        
        # Add loading spinner during model response
        submit_btn.click(
            get_response, 
            inputs=input_box, 
            outputs=output_box, 
            show_progress=True
        )
         # Clear button functionality
        clear_btn.click(
            lambda: ("", ""), inputs=None, outputs=[input_box, output_box]
        )
           
    gr.Markdown("---")
    gr.HTML(
        '<div style="text-align: center;"><a href="https://github.com/JunoLeong/DocExtractRAG"> Juno Leong © 2024 DocExtractRAG </a></div>'
    )
iface.launch()