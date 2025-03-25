from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
import gradio as gr
import base64
from pathlib import Path

# Load environment variables
load_dotenv()

# Paths for storing downloaded models
BASE_DIR = Path("models")
LLM_MODEL_PATH = BASE_DIR / "llms"
EMBED_MODEL_PATH = BASE_DIR / "embeddings"

# Ensure the base directory exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Models to be used
LLM_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EMBED_MODEL_NAME = "BAAI/bge-large-en"

# Download models if not already present
def download_model(model_name, destination):
    #Download a model from Hugging Face and save it locally.
    if not destination.exists():
        print(f"Downloading {model_name} to {destination}...")
        snapshot_download(repo_id=model_name, local_dir=destination)
        print(f"Model {model_name} downloaded to {destination}.")
    else:
        print(f"Model {model_name} already exists at {destination}.")

download_model(EMBED_MODEL_NAME, EMBED_MODEL_PATH)

vector_index = None

# Initialize the parser
parser = LlamaParse(api_key=os.getenv("LLAMA_INDEX_API"), result_type='markdown')

# Define file extractor with common extensions
file_extractor = {
    '.pdf': parser,
    '.docx': parser,
    '.doc': parser,
    '.txt': parser,
    '.csv': parser,
    '.xlsx': parser,
    '.pptx': parser,
    '.html': parser,
    '.jpg': parser,
    '.jpeg': parser,
    '.png': parser,
    '.webp': parser,
    '.svg': parser,
}

# File processing function
def load_files(file_path: str):
    #Load files, create an embedding, and initialize a vector index.
    try:
        global vector_index
        document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
        embed_model = HuggingFaceEmbedding(model_name=str(EMBED_MODEL_PATH))
        vector_index = VectorStoreIndex.from_documents(document, embed_model=embed_model)
        print(f"Parsing done for {file_path}")
        filename = os.path.basename(file_path)
        return f"Ready to give response on {filename}"
    except Exception as e:
        return f"An error occurred: {e}"

# Respond function
# Respond function that uses the globally set selected model
def respond(message, history):
    try:
        # Initialize the LLM with the selected model
        llm = HuggingFaceInferenceAPI(
            model_name=LLM_MODEL_NAME,
            contextWindow=8192,  # Context window size (typically max length of the model)
            maxTokens=1024,  # Tokens per response generation (512-1024 works well for detailed answers)
            temperature=0.3,  # Lower temperature for more focused answers (0.2-0.4 for factual info)
            topP=0.9,  # Top-p sampling to control diversity while retaining quality
            frequencyPenalty=0.5,  # Slight penalty to avoid repetition
            presencePenalty=0.5,  # Encourages exploration without digressing too much
            token=os.getenv("TOKEN")
        )

        # Set up the query engine with the selected LLM
        query_engine = vector_index.as_query_engine(llm=llm)
        bot_message = query_engine.query(message)

        print(f"\n{datetime.now()}:{LLM_MODEL_NAME}:: {message} --> {str(bot_message)}\n")
        return f"{LLM_MODEL_NAME}:\n{str(bot_message)}"
    except Exception as e:
        if str(e) == "'NoneType' object has no attribute 'as_query_engine'":
            return "Please upload a file."
        return f"An error occurred: {e}"

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode the images
github_logo_encoded = encode_image("Images/github-logo.png")
linkedin_logo_encoded = encode_image("Images/linkedin-logo.png")
website_logo_encoded = encode_image("Images/ai-logo.png")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# DocBot📄🤖")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="single", type='filepath', label="Step-1: Upload document")
            btn = gr.Button("Submit", variant='primary')
            output = gr.Text(label='Vector Index')
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(placeholder="Step-2: Ask me questions on the uploaded document!", container=False)
            )
    btn.click(fn=load_files, inputs=[file_input], outputs=output)

# Launch the app
if __name__ == "__main__":
    demo.launch()