description = '''
# 📄 **Document QA Bot: A RAG-Based Application for Interactive Document Querying**
Welcome to the Document QA Bot, a sophisticated Retrieval-Augmented Generation (RAG) application that utilizes **LlamaIndex** and **Hugging Face** models to answer questions based on documents you upload. This bot is designed to empower you with rapid, insightful responses, providing a choice of language models (LLMs) and embedding models that cater to various requirements, including performance, accuracy, and response time.
## ✨ **Application Overview**
With Document QA Bot, you can interactively query your document, receive contextual answers, and dynamically switch between LLMs as needed for optimal results. The bot supports various file formats, allowing you to upload and analyze different types of documents and even some image formats. 
### **Key Features**
- **Choice of Models:** Access a list of powerful LLMs and embedding models for optimal results.
- **Flexible Document Support:** Multiple file types supported, including images.
- **Real-time Interaction:** Easily switch between models for experimentation and fine-tuning answers.
- **User-Friendly:** Seamless experience powered by Gradio's intuitive interface.
---
## 🚀 **Steps to Use the Document QA Bot**
1. **Upload Your File**  
   Begin by uploading a document. Supported formats include `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`, `.pptx`, `.html`, `.jpg`, `.png`, and more.
   
2. **Select Embedding Model**  
   Choose an embedding model to parse and index the document’s contents, then submit. Wait for the confirmation message that the document has been successfully indexed.
3. **Choose a Language Model (LLM)**  
   Pick an LLM from the dropdown to tailor the bot’s response style and accuracy.
4. **Start Chatting**  
   Ask questions about your document! You can switch between LLMs as needed for different insights or to test model behavior on the same question.
---
## ⚙️ **How the Application Works**
Upon uploading a document, the bot utilizes **LlamaParse** to parse its content. The parsed data is then indexed with a selected embedding model, generating a vector representation that enables quick and relevant responses. When you ask questions, the chosen LLM interprets the document context to generate responses specific to the content uploaded.
---
## 🔍 **Available LLMs and Embedding Models**
### **Embedding Models** (For indexing document content)
1. **`BAAI/bge-large-en`**  
   - **Size**: 335M parameters  
   - **Best For**: Complex, detailed embeddings; slower but yields high accuracy.
2. **`BAAI/bge-small-en-v1.5`**  
   - **Size**: 33.4M parameters  
   - **Best For**: Faster embeddings, ideal for lighter workloads and quick responses.
3. **`NeuML/pubmedbert-base-embeddings`**  
   - **Size**: 768-dimensional dense vector space  
   - **Best For**: Biomedical or medical-related text; highly specialized.
4. **`BAAI/llm-embedder`**  
   - **Size**: 109M parameters  
   - **Best For**: Basic embeddings for straightforward use cases.
### **LLMs** (For generating answers)
1. **`mistralai/Mixtral-8x7B-Instruct-v0.1`**  
   - **Size**: 46.7B parameters
   - **Purpose**: Demonstrates compelling performance with minimal fine-tuning. Suited for unmoderated or exploratory use.
2. **`meta-llama/Meta-Llama-3-8B-Instruct`**  
   - **Size**: 8.03B parameters
   - **Purpose**: Optimized for dialogue, emphasizing safety and helpfulness. Excellent for structured, instructive responses.
3. **`mistralai/Mistral-7B-Instruct-v0.2`**  
   - **Size**: 7.24B parameters
   - **Purpose**: Fine-tuned for effectiveness; lacks moderation, useful for quick demonstration purposes.
4. **`tiiuae/falcon-7b-instruct`**  
   - **Size**: 7.22B parameters
   - **Purpose**: Robust open-source model for inference, leveraging large-scale data for highly contextual responses.
---
## 🔗 **Best Embedding Model Combinations for Optimal Performance in RAG**
The choice of embedding models plays a crucial role in determining the speed and accuracy of document responses. Since you can dynamically switch LLMs during the chat, focusing on an optimal embedding model at the outset will significantly influence response quality and efficiency. Below is a guide to the best embedding models for various scenarios based on the need for time efficiency and answer accuracy.
| **Scenario**                  | **Embedding Model**                  | **Strengths**                                      | **Trade-Offs**                       |
|:-----------------------------:|:------------------------------------:|:--------------------------------------------------:|:------------------------------------:|
| **Fastest Response**          | `BAAI/bge-small-en-v1.5`            | Speed-oriented, ideal for high-frequency querying  | May miss nuanced details             |
| **High Accuracy for Large Texts** | `BAAI/bge-large-en`               | High accuracy, captures complex document structure | Slower response time                |
| **Balanced General Purpose**  | `BAAI/llm-embedder` | Reliable, quick response, adaptable across topics | Moderate accuracy, general use case  |
| **Biomedical & Specialized Text** | `NeuML/pubmedbert-base-embeddings` | Optimized for medical and scientific text          | Specialized, slightly slower         |
---
## 📂 **Supported File Formats**
The bot supports a range of document formats, making it versatile for various data sources. Below are the currently supported formats:
- **Documents**: `.pdf`, `.docx`, `.doc`, `.txt`, `.csv`, `.xlsx`, `.pptx`, `.html`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.svg`
---
## 🎯 **Use Cases**
1. **Educational Research**  
   Upload research papers or study materials and get precise answers for revision or note-taking.
2. **Corporate Data Analysis**  
   Interrogate reports, presentations, or financial data for quick insights without reading extensive documents.
3. **Legal Document Analysis**  
   Analyze lengthy legal documents by querying clauses, terms, and specific details.
4. **Healthcare and Scientific Research**  
   Access detailed insights into medical or scientific documents with models trained on domain-specific data.
---
### 🌟 **Get Started Today and Experience Document-Centric Question Answering**  
Whether you're a student, researcher, or professional, the Document QA Bot is your go-to tool for interactive, accurate document analysis. Upload your file, select your model, and dive into a seamless question-answering experience tailored to your document's unique content.
'''

guide = '''
### Embedding Models and Trade-Offs
| **Embedding Model**         | **Speed (Vector Index)** | **Advantages**                      | **Trade-Offs**                  |
|-----------------------------|-------------------|-------------------------------------|---------------------------------|
| `BAAI/bge-small-en-v1.5`    | **Fastest**       | Ideal for quick indexing            | May miss nuanced details        |
| `BAAI/llm-embedder`         | **Fast**          | Balanced performance and detail     | Slightly less precise than large models |
| `BAAI/bge-large-en`         | **Slow**          | Best overall precision and detail   | Slower due to complexity        |
    
### Language Models (LLMs) and Use Cases
| **LLM**                             | **Best Use Case**                       |
|------------------------------------|-----------------------------------------|
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Works well for **both short and long answers** |
| `meta-llama/Meta-Llama-3-8B-Instruct`  | Ideal for **long-length answers**         |
| `tiiuae/falcon-7b-instruct`           | Best suited for **short-length answers**  |
'''

footer = """
<div style="background-color: #1d2938; color: white; padding: 10px; width: 100%; bottom: 0; left: 0; display: flex; justify-content: space-between; align-items: center; padding: .2rem 35px; box-sizing: border-box; font-size: 16px;">
    <div style="text-align: left;">
        <p style="margin: 0;">&copy; 2024 </p>
    </div>
    <div style="text-align: center; flex-grow: 1;">
        <p style="margin: 0;">      This website is made with ❤ by SARATH CHANDRA</p>
    </div>
    <div class="social-links" style="display: flex; gap: 20px; justify-content: flex-end; align-items: center;">
        <a href="https://github.com/21bq1a4210" target="_blank" style="text-align: center;">
            <img src="data:image/png;base64,{}" alt="GitHub" width="40" height="40" style="display: block; margin: 0 auto;">
            <span style="font-size: 14px;">GitHub</span>
        </a>
        <a href="https://www.linkedin.com/in/sarath-chandra-bandreddi-07393b1aa/" target="_blank" style="text-align: center;">
            <img src="data:image/png;base64,{}" alt="LinkedIn" width="40" height="40" style="display: block; margin: 0 auto;">
            <span style="font-size: 14px;">LinkedIn</span>
        </a>
        <a href="https://21bq1a4210.github.io/MyPortfolio-/" target="_blank" style="text-align: center;">
            <img src="data:image/png;base64,{}" alt="Portfolio" width="40" height="40" style="display: block; margin-right: 40px;">
            <span style="font-size: 14px;">Portfolio</span>
        </a>
    </div>
</div>
"""