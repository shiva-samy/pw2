from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
import gradio as gr
from pathlib import Path

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

def download_model(model_name, destination):
    if not destination.exists():
        print(f"Downloading {model_name} to {destination}...")
        snapshot_download(repo_id=model_name, local_dir=destination)
        print(f"Model {model_name} downloaded to {destination}.")
    else:
        print(f"Model {model_name} already exists at {destination}.")

download_model(EMBED_MODEL_NAME, EMBED_MODEL_PATH)

# Global variables
vector_index = None
parser = LlamaParse(api_key=os.getenv("LLAMA_INDEX_API"), result_type='markdown')
file_extractor = {'.pdf': parser, '.docx': parser, '.doc': parser, '.txt': parser, '.csv': parser, '.xlsx': parser, '.pptx': parser, '.html': parser, '.jpg': parser, '.jpeg': parser, '.png': parser, '.webp': parser, '.svg': parser}

def load_files(file_paths):
    try:
        global vector_index
        # Load all uploaded documents
        documents = []
        for file_path in file_paths:
            document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
            documents.extend(document)
            print(f"Parsing done for {file_path}")

        # Create a single VectorStoreIndex from all documents
        embed_model = HuggingFaceEmbedding(model_name=str(EMBED_MODEL_PATH))
        if vector_index is None:
            vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        else:
            # Update the existing index with new documents
            for doc in documents:
                vector_index.insert(doc)
        print("All documents processed and indexed.")

        # Return status message
        filenames = [os.path.basename(file_path) for file_path in file_paths]
        return f"Ready to chat about: {', '.join(filenames)}"
    except Exception as e:
        return f"An error occurred: {e}"

def respond(message, history):
    try:
        if vector_index is None:
            return history + [(message, "Please upload documents first.")]

        # Initialize the LLM
        llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=os.getenv("TOKEN"))

        # Query the combined VectorStoreIndex
        query_engine = vector_index.as_query_engine(llm=llm)
        bot_message = query_engine.query(message)

        # Check if the response is relevant to the documents
        if "I don't know" in str(bot_message) or "not found" in str(bot_message).lower():
            bot_message = "I don't have information about that in the uploaded documents."

        print(f"\n{datetime.now()}:{LLM_MODEL_NAME}:: {message} --> {str(bot_message)}\n")

        # Append the user's message and bot's response to the history
        history.append((message, str(bot_message)))

        # Return the updated history
        return history
    except Exception as e:
        error_message = "Please upload documents first." if "'NoneType' object has no attribute 'as_query_engine'" in str(e) else f"An error occurred: {e}"
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

def generate_report(prompt):
    try:
        if vector_index is None:
            return "Please upload documents first."

        # Initialize the LLM and query engine
        llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=os.getenv("TOKEN"))
        query_engine = vector_index.as_query_engine(llm=llm)

        # Generate the report structure based on the user's prompt
        report_structure = llm.complete(f"Generate a report structure based on the following prompt: {prompt}").text
        print(f"Report Structure: {report_structure}")

        # Generate content for each section in the report structure
        report_content = ""
        for section in report_structure.split("\n"):
            if section.strip():
                query = f"Provide detailed information for the section: {section}"
                answer = str(query_engine.query(query))
                report_content += f"# {section}\n\n{answer}\n\n"

        # Generate the Markdown report
        return generate_markdown_report(report_content)
    except Exception as e:
        return f"Error generating report: {e}"

def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# DocBot📄🤖 - Document Analysis Tool")
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(file_count="multiple", type='filepath', label="Upload document(s)")
                btn = gr.Button("Submit", variant='primary', interactive=False)
                output = gr.Text(label='Status')
                chat_button = gr.Button("Open Chat")
                report_button = gr.Button("Open Report Generator")
        
        # Define chat and report interfaces
        with gr.Column(visible=False) as chat_interface:
            chatbot = gr.Chatbot(height=500)
            textbox = gr.Textbox(placeholder="Ask me anything about the uploaded documents!", container=False)
            chat_interface_visible = gr.Checkbox(value=False, visible=False)
        
        with gr.Column(visible=False) as report_interface:
            report_prompt = gr.Textbox(label="Enter the report structure (e.g., 'Generate a report with sections for personal information, education, and skills')")
            generate_button = gr.Button("Generate Report")
            report_output = gr.Textbox(label="Generated Report", interactive=False)
            report_interface_visible = gr.Checkbox(value=False, visible=False)
        
        # Enable Submit button when files are selected
        def enable_submit(file_paths):
            return gr.update(interactive=bool(file_paths))

        file_input.change(fn=enable_submit, inputs=[file_input], outputs=[btn])

        # Button actions
        btn.click(fn=load_files, inputs=[file_input], outputs=output)
        
        def toggle_chat_interface():
            return gr.update(visible=True), gr.update(visible=False)
        
        def toggle_report_interface():
            return gr.update(visible=False), gr.update(visible=True)
        
        chat_button.click(fn=toggle_chat_interface, outputs=[chat_interface, report_interface])
        report_button.click(fn=toggle_report_interface, outputs=[chat_interface, report_interface])
        
        # Chat interface
        textbox.submit(fn=respond, inputs=[textbox, chatbot], outputs=[chatbot])
        
        # Report interface
        generate_button.click(fn=generate_report, inputs=[report_prompt], outputs=[report_output])
    
    demo.launch()

if __name__ == "__main__":
    main_interface()