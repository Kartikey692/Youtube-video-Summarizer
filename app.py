import re
import os  
import uvicorn
from fastapi import FastAPI
import gradio as gr
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain_mistralai import ChatMistralAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

app = FastAPI()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "youtube-summarizer"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Initialize Mistral AI
chat_model = ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="open-mistral-7b",
    temperature=0.7
)

# Initialize Embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    api_key=os.getenv("HF_API_KEY")
)

# Define Summary Prompt
summary_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    You are an expert summarizer. Given the YouTube video transcript, generate a structured summary:
    
    1. **Main Topic**
    2. **Key Points**
    3. **Important Details**
    4. **Conclusion**
    
    Transcript:
    {context}
    """
)

# Function to extract video ID
def extract_youtube_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to get transcript
def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join((t["text"] for t in transcript_list))
        return transcript
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# Function to store transcript in Pinecone
def vector_store_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(transcript)

    vector_store = VectorStorePinecone.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vector_store

# Setup QA chain
def setup_qa_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return qa_chain

# Generate Summary
def generate_video_summary(transcript):
    prompt_text = summary_prompt.format(context=transcript)
    response = chat_model.invoke(prompt_text)
    return response.content

# Gradio UI
def process_video_ui(video_url):
    video_id = extract_youtube_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL", ""

    transcript = get_video_transcript(video_id)
    if "Error" in transcript:
        return transcript, ""

    vector_store = vector_store_transcript(transcript)
    summary = generate_video_summary(transcript)
    global qa_chain
    qa_chain = setup_qa_chain(vector_store)

    return summary, "Processing complete! You can now ask questions."

def ask_question_ui(query):
    if "qa_chain" not in globals():
        return "Process a video first to enable Q&A."
    response = qa_chain.invoke({"question": query})
    return response["answer"] if "answer" in response else "No answer found."

gr_interface = gr.Blocks()

with gr_interface:
    gr.Markdown("## 🎥 YouTube Video Summarizer & Q&A")
    with gr.Row():
        video_input = gr.Textbox(label="🔗 Enter YouTube Video URL", placeholder="Paste your video link here...")
        process_button = gr.Button("📄 Process Video")
    
    summary_output = gr.Textbox(label="📜 Summary", interactive=False)
    status_output = gr.Textbox(label="✅ Status", interactive=False)
    process_button.click(process_video_ui, inputs=video_input, outputs=[summary_output, status_output])

    gr.Markdown("### ❓ Ask a Question About the Video")
    with gr.Row():
        question_input = gr.Textbox(label="📝 Your Question", placeholder="Type your question here...")
        qa_button = gr.Button("🤖 Get Answer")
    
    answer_output = gr.Textbox(label="💡 Answer", interactive=False)
    qa_button.click(ask_question_ui, inputs=question_input, outputs=answer_output)

# Run FastAPI with Gradio
app = gr.mount_gradio_app(app, gr_interface, path="/")

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8000))         # Use Render's port or default to 8000
    host = os.environ.get("HOST", "127.0.0.1")        # Use 127.0.0.1 locally, override to 0.0.0.0 on Render

    uvicorn.run(app, host=host, port=port)

