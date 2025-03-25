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
import sqlite3

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

# SQLite database path
SQLITE_DB_PATH = "chat_sessions.db"

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Create files table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create chat_sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create messages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        file_id INTEGER,
        sender_type TEXT NOT NULL,
        message_type TEXT NOT NULL,
        text_content TEXT,
        blob_content BLOB,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id),
        FOREIGN KEY (file_id) REFERENCES files(file_id)
    );
    """)

    conn.commit()
    conn.close()

def get_db_connection():
    """Get a connection to the SQLite database."""
    return sqlite3.connect(SQLITE_DB_PATH)

def save_file_metadata(file_name, file_path):
    """Save file metadata to the database and return the file_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO files (file_name, file_path) VALUES (?, ?)",
        (file_name, file_path)
    )
    conn.commit()
    file_id = cursor.lastrowid
    conn.close()
    return file_id

def save_chat_message(session_id, file_id, sender_type, message_type, text_content=None, blob_content=None):
    """Save a chat message to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, file_id, sender_type, message_type, text_content, blob_content) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, file_id, sender_type, message_type, text_content, blob_content)
    )
    conn.commit()
    conn.close()

def load_chat_history(session_id):
    """Load chat history from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sender_type, message_type, text_content, blob_content FROM messages WHERE session_id = ?",
        (session_id,)
    )
    history = cursor.fetchall()
    conn.close()
    return history

def create_chat_session(session_name):
    """Create a new chat session and return the session_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (session_name) VALUES (?)",
        (session_name,)
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def rename_chat_session(session_id, new_name):
    """Rename a chat session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET session_name = ? WHERE session_id = ?",
        (new_name, session_id))
    conn.commit()
    conn.close()

def get_chat_sessions():
    """Retrieve all chat sessions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, session_name FROM chat_sessions ORDER BY created_at DESC")
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# Initialize the database
init_db()

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

            # Save file metadata to the database
            file_name = os.path.basename(file_path)
            file_id = save_file_metadata(file_name, file_path)
            print(f"File metadata saved with ID: {file_id}")

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

def respond(message, history, session_id):
    try:
        if not session_id:  # If no session is selected, create a new one
            session_id = create_chat_session("New Chat")
            session_name = f"Chat {session_id}"  # Default name for the new chat
            rename_chat_session(session_id, session_name)

        # Query Qdrant for relevant information
        relevant_info = query_qdrant(message)

        if not relevant_info:
            bot_message = "No relevant information found."
        else:
            # Initialize the LLM
            llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=os.getenv("TOKEN"))

            # Generate a response based on the relevant information
            bot_message = llm.complete(f"Based on the following information, answer the query: {message}\n\n{relevant_info}").text

        print(f"\n{datetime.now()}:{LLM_MODEL_NAME}:: {message} --> {str(bot_message)}\n")

        # Save the user's message and bot's response to the database
        save_chat_message(session_id, None, "user", "text", message)
        save_chat_message(session_id, None, "bot", "text", bot_message)

        # Append the user's message and bot's response to the history
        history.append((message, str(bot_message)))

        # Return the updated history and session ID
        return history, session_id
    except Exception as e:
        error_message = f"An error occurred: {e}"
        history.append((message, error_message))
        return history, session_id

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
        # Query Qdrant for relevant information
        relevant_info = query_qdrant(prompt)

        if not relevant_info:
            return "No relevant information found."

        # Initialize the LLM
        llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=os.getenv("TOKEN"))

        # Generate a structured report
        report_structure = llm.complete(
            f"Generate a report structure about {prompt} based on the following information: {relevant_info}"
        ).text
        print(f"Report Structure: {report_structure}")

        # Generate content for each section in the report structure
        report_content = ""
        for section in report_structure.split("\n"):
            if section.strip():
                # Query Qdrant for relevant information for this section
                section_info = query_qdrant(section)
                if section_info:
                    report_content += f"# {section}\n\n{section_info[0]}\n\n"
                else:
                    report_content += f"# {section}\n\nNo relevant information found.\n\n"

        # Generate the Markdown report
        return generate_markdown_report(report_content)
    except Exception as e:
        return f"Error generating report: {e}"

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
                with gr.Row():
                    with gr.Column(scale=1):
                        # Initialize the dropdown with existing chat sessions
                        sessions = get_chat_sessions()
                        chat_sessions = gr.Dropdown(choices=[session[1] for session in sessions], label="Chat Sessions", interactive=True)
                        new_chat_btn = gr.Button("New Chat", interactive=not sessions)  # Disable if sessions exist
                        rename_chat_btn = gr.Button("Rename Chat", interactive=bool(sessions))  # Enable only if sessions exist
                        rename_textbox = gr.Textbox(label="New Chat Name", placeholder="Enter new chat name", interactive=bool(sessions))  # Enable only if sessions exist
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500)
                        textbox = gr.Textbox(placeholder="Ask me anything about the uploaded documents!", container=False)
                        textbox.submit(fn=respond, inputs=[textbox, chatbot, chat_sessions], outputs=[chatbot])

                # Function to handle new chat creation
                def create_new_chat():
                    session_id = create_chat_session("New Chat")
                    sessions = get_chat_sessions()
                    return [session[1] for session in sessions], [], gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)

                new_chat_btn.click(fn=create_new_chat, outputs=[chat_sessions, chatbot, new_chat_btn, rename_chat_btn, rename_textbox])

                # Function to handle chat renaming
                def rename_chat(session_name, new_name):
                    if not session_name:  # If no session is selected, return the current list of sessions
                        return [session[1] for session in get_chat_sessions()]
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT session_id FROM chat_sessions WHERE session_name = ?",
                        (session_name,)
                    )
                    session_id = cursor.fetchone()[0]
                    rename_chat_session(session_id, new_name)
                    return [session[1] for session in get_chat_sessions()]

                rename_chat_btn.click(fn=rename_chat, inputs=[chat_sessions, rename_textbox], outputs=[chat_sessions])

                # Function to load chat history when a session is selected
                def load_chat(session_name):
                    if not session_name:  # If no session is selected, return an empty history
                        return []
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT session_id FROM chat_sessions WHERE session_name = ?",
                        (session_name,)
                    )
                    session_id = cursor.fetchone()[0]
                    history = load_chat_history(session_id)
                    conn.close()
                    return history

                chat_sessions.change(fn=load_chat, inputs=[chat_sessions], outputs=[chatbot])

            # Report Generator Tab
            with gr.Tab("Report Generator"):
                report_prompt = gr.Textbox(label="Enter the report structure (e.g., 'Generate a report with sections for personal information, education, and skills')")
                generate_button = gr.Button("Generate Report")
                report_output = gr.Textbox(label="Generated Report", interactive=False)
                generate_button.click(fn=generate_report, inputs=[report_prompt], outputs=[report_output])

    demo.launch()

if __name__ == "__main__":
    main_interface()