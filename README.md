# **DocExtractRAG**

## Overview 🌟
**DocExtractRAG** is a Retrieval-Augmented Generation (RAG) system that combines the power of large language models (LLMs) with document retrieval to provide insightful responses based on academic or other types of documents. The DocExtractRAG utilizes the **Zephyr-7B-beta** model for text generation, **BAAI/bge-large-en** for document embeddings, and **Chroma** for efficient document retrieval.

Users can interact with the DocExtractRAG via a **Gradio** web interface, where they input a question, and the DocExtractRAG retrieves relevant information from the stored documents and generates a concise, structured answer.

## System Components 🧩
<div align="center">
<table border="1" style="border-collapse: collapse; width: 60%;">
  <thead>
    <tr>
      <th><strong>Components</strong></th>
      <th><strong>Details</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Model Used</strong></td>
      <td><a href="https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF">zephyr-7B-beta-GGUF</a></td>
    </tr>
    <tr>
      <td><strong>Embeddings Model</strong></td>
      <td>BAAI/bge-large-en</td>
    </tr>
    <tr>
      <td><strong>Vector Store</strong></td>
      <td>Chroma</td>
    </tr>
    <tr>
      <td><strong>Accelerator</strong></td>
      <td>CTransformers</td>
    </tr>
    <tr>
      <td><strong>Framework</strong></td>
      <td>LangChain</td>
    </tr>
    <tr>
      <td><strong>Interface</strong></td>
      <td>Gradio</td>
    </tr>
  </tbody>
</table>
</div>

## System Setup 🛠️
To get started with **DocExtractRAG**, follow these steps:
 
1. Clone the repository and install the required libraries: 
      ```bash 
   pip install -r requirements.txt
   ````

2. Download the Zephyr-7B-beta model using the Hugging Face CLI:
   ``` bash
   huggingface-cli download TheBloke/zephyr-7B-beta-GGUF zephyr-7b-beta.Q4_K_M.gguf --local-dir .\models\zephyr-7B-beta-GGUF --local-dir-use-symlinks False 
   ```
3. For optimal performance on GPUs, install ctransformers with CUDA acceleration:
   ``` bash
   pip install ctransformers[cuda]
   ```

4. Initialize the vector store by running the vector_store.py script:
   ```bash
   python vector_store.py
   ```
  >  **Tip**: If you want to process a different PDF document, update the file name in line 28 of vector_store.py: <br>
      ```
      loader = PyPDFLoader("file_name")
      ```

5. Launch the application:
   ``` bash
   python app.py
   ```

6. Running the DocExtractRAG in your local host.  `http://127.0.0.1:7860`
   
**Note:**
> - Check for breaking changes and deprecations for Langchain libraires (Optional): ```langchain-cli migrate --diff app.py``` 

> - If CUDA error due to CUDA driver version is insufficient for CUDA runtime version: Update your CUDA toolkits version on [Nvidia Developer Website - cuda download](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) 

## About the Interface 🌈
The Gradio interface enables intuitive interaction with DocExtractRAG.
- gradio.Blocks 🧱: This low-level API allows full control over data flows and layout. To build a more complex and customizable application interface, `Blocks is used instead of gradio.interface`.

### Here is the user interface of DocExtractRAG:
<img src="Documentation_img\UI_gradio_DocExtractRAG.png" alt = 'User Interface DocExtractRAG'>

## Reference 📚
- [LLM Model](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) <br>
  *Citation*: **Tunstall, Lewis et al. (2023)**. *Zephyr: Direct Distillation of LM Alignment*. [arXiv:2310.16944](https://arxiv.org/abs/2310.16944)

- [Exploring offline RAG with Langchain, Zephyr-7b-beta and DeciLM-7b](https://medium.com/aimonks/exploring-offline-rag-with-langchain-zephyr-7b-beta-and-decilm-7b-c0626e09ee1f)

- [Gradio Documentation](https://huggingface.co/learn/nlp-course/chapter9/7)

---
Feel free to reach out with suggestions or improvements for the project! If you like or are using this project, please consider giving it a star⭐. Thanks! (●'◡'●)
