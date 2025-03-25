from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import snapshot_download
import os, time
from dotenv import load_dotenv
import gradio as gr
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()

# Paths for storing downloaded models and reports
BASE_DIR = Path("models")
REPORTS_DIR = Path("reports")
LLM_MODEL_PATH = BASE_DIR / "llms"
EMBED_MODEL_PATH = BASE_DIR / "embeddings"

# Ensure the base directories exist
BASE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to be used
LLM_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EMBED_MODEL_NAME = "BAAI/bge-large-en"

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "pw2-vdb"

def download_model(model_name, destination):
    if not destination.exists():
        print(f"Downloading {model_name} to {destination}...")
        snapshot_download(repo_id=model_name, local_dir=destination)
        print(f"Model {model_name} downloaded to {destination}.")
    else:
        print(f"Model {model_name} already exists at {destination}.")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

# Create or load Qdrant collection
try:
    qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
except:
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  # Adjust size based on your embedding model
    )
    print("Database didn't exist, so created pw2-vdb!")

# Global variables
parser = LlamaParse(api_key=os.getenv("LLAMA_INDEX_API"), result_type='markdown')
file_extractor = {'.pdf': parser, '.docx': parser, '.doc': parser, '.txt': parser, '.csv': parser, '.xlsx': parser, '.pptx': parser, '.html': parser, '.jpg': parser, '.jpeg': parser, '.png': parser, '.webp': parser, '.svg': parser}

def load_files(file_paths):
    try:
        documents = []
        for file_path in file_paths:
            start_time = time.time()
            document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
            documents.extend(document)
            print(f"Parsing done for {file_path} in {time.time() - start_time} seconds")

        embed_model = HuggingFaceEmbedding(model_name=str(EMBED_MODEL_PATH))
        start_time = time.time()
        embeddings = embed_model.get_text_embedding_batch([doc.text for doc in documents])
        print(f"Embedding generation time: {time.time() - start_time} seconds")

        # Store embeddings in Qdrant
        start_time = time.time()
        for doc, embedding in zip(documents, embeddings):
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    {
                        "id": doc.doc_id,
                        "vector": embedding,
                        "payload": {"text": doc.text, "metadata": doc.metadata}
                    }
                ]
            )
        print(f"Qdrant upsert time: {time.time() - start_time} seconds")

        filenames = [os.path.basename(file_path) for file_path in file_paths]
        return f"Ready to chat about: {', '.join(filenames)}"
    except Exception as e:
        return f"An error occurred: {e}"

def query_qdrant(query_text, top_k=5):
    """Query Qdrant database for relevant information."""
    embed_model = HuggingFaceEmbedding(model_name=str(EMBED_MODEL_PATH))
    query_embedding = embed_model.get_text_embedding(query_text)

    # Search Qdrant for relevant documents
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    # Extract relevant information from search results
    relevant_info = []
    for result in search_results:
        relevant_info.append(result.payload["text"])

    return relevant_info

def generate_report(prompt):
    try:
        # Query Qdrant for relevant information
        relevant_info = query_qdrant(prompt)

        if not relevant_info:
            return "No relevant information found."

        # Initialize the LLM
        llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=os.getenv("TOKEN"))

        # Generate the entire report in one go
        report_prompt = f"""
        {prompt}.
        Use the following information as context:
        {relevant_info}

        If structure of the report is mentioned, then follow that structure.
        Else, Structure the report as follows:
        - Title: A concise title for the report.
        - Introduction: A brief overview of the topic.
        - Analysis: Detailed insights and findings.
        - Conclusion: Summary and recommendations.

        Ensure the report is comprehensive and avoids repetition also try to maintain the size of the report if mentioned.
        """
        report_content = llm.complete(report_prompt, max_new_tokens=2000).text

        # Save the report
        return generate_markdown_report(report_content)
    except Exception as e:
        return f"Error generating report: {e}"

def respond(message, history, response_length="medium"):
    try:
        # Query Qdrant for relevant information
        relevant_info = query_qdrant(message)

        if not relevant_info:
            bot_message = "No relevant information found."
        else:
            # Initialize the LLM
            llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=os.getenv("TOKEN"))

            # Generate a response based on the desired length
            if response_length == "short":
                bot_message = llm.complete(
                    f"Provide a concise answer to the query: {message}\n\n{relevant_info}",
                    max_new_tokens=200
                ).text
            elif response_length == "medium":
                bot_message = llm.complete(
                    f"Provide a detailed answer to the query: {message}\n\n{relevant_info}",
                    max_new_tokens=500
                ).text
            elif response_length == "long":
                bot_message = llm.complete(
                    f"Provide a comprehensive answer to the query: {message}\n\n{relevant_info}",
                    max_new_tokens=1000
                ).text

        print(f"\n{datetime.now()}:{LLM_MODEL_NAME}:: {message} --> {str(bot_message)}\n")

        # Append the user's message and bot's response to the history
        history.append((message, str(bot_message)))

        # Return the updated history
        return history
    except Exception as e:
        error_message = f"An error occurred: {e}"
        history.append((message, error_message))
        return history

def generate_markdown_report(report_content):
    """Generate a Markdown report."""
    # Create the report content
    markdown_content = f"# Generated Report\n\n{report_content}"

    # Save the Markdown file
    report_filename = REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return f"Report generated and saved at {report_filename}"

def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# DocBot📄🤖 - Document Analysis Tool")

        # Tabs for Upload, Chatbot, and Report Generator
        with gr.Tabs():
            # Upload Tab
            with gr.Tab("Upload Documents"):
                file_input = gr.File(file_count="multiple", type='filepath', label="Upload document(s)")
                btn = gr.Button("Submit", variant='primary', interactive=False)
                output = gr.Text(label='Upload Status')

                # Enable Submit button when files are selected
                def enable_submit(file_paths):
                    return gr.update(interactive=bool(file_paths))

                file_input.change(fn=enable_submit, inputs=[file_input], outputs=[btn])

                # Button actions
                btn.click(fn=load_files, inputs=[file_input], outputs=output)

            # Chatbot Tab
            with gr.Tab("Chatbot"):
                chatbot = gr.Chatbot(height=500)
                textbox = gr.Textbox(placeholder="Ask me anything about the uploaded documents!", container=False)
                textbox.submit(fn=respond, inputs=[textbox, chatbot], outputs=[chatbot])

            # Report Generator Tab
            with gr.Tab("Report Generator"):
                report_prompt = gr.Textbox(label="Enter the report structure (e.g., 'Generate a report with sections for personal information, education, and skills')")
                generate_button = gr.Button("Generate Report")
                report_output = gr.Textbox(label="Generated Report", interactive=False)
                generate_button.click(fn=generate_report, inputs=[report_prompt], outputs=[report_output])

    demo.launch()

if __name__ == "__main__":
    main_interface()