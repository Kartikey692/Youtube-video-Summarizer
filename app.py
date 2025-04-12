import re
import os
import uvicorn
from fastapi import FastAPI
import gradio as gr
from dotenv import load_dotenv
from langdetect import detect
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from langchain_mistralai import ChatMistralAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI()

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "youtube-summarizer"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Mistral AI
chat_model = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="open-mistral-7b",
    temperature=0.7
)

# Embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    api_key=HF_API_KEY
)

# Prompt Template
summary_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a professional summarizer with expertise in simplifying video transcripts into clean, structured summaries. Read the transcript and provide:

##  Main Idea
Summarize the core topic or theme of the video.

## Key Takeaways
List the 3–5 main points discussed.

## Important Details
Highlight critical sub-points or supporting information.

## Conclusion
Provide a brief conclusion that wraps up the summary.

Here’s the transcript:
{context}
"""
)

# Extract Video ID
def extract_youtube_video_id(url):
    patterns = [r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Get Transcript and Translate if needed
def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
            raw_text = " ".join([t['text'] for t in transcript.fetch()])
            return raw_text
        except:
            for t in transcript_list:
                if t.is_generated and t.is_translatable:
                    translated = t.translate('en')
                    raw_text = " ".join([item['text'] for item in translated.fetch()])
                    return raw_text
        return "No English or translatable transcripts available."
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return f"Transcript error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Vector Store
def vector_store_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript)
    vector_store = VectorStorePinecone.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    return vector_store

# QA Chain Setup
def setup_qa_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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

# Gradio UI Functions
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
    return response.get("answer", "No answer found.")

# Gradio Interface
gr_interface = gr.Blocks()

with gr_interface:
    gr.Markdown("## 🎥 YouTube Video Summarizer + Q&A")
    with gr.Row():
        video_input = gr.Textbox(label="🔗 YouTube URL", placeholder="Paste your video link...")
        process_button = gr.Button("📄 Process Video")

    summary_output = gr.Textbox(label="📜 Summary", lines=10, interactive=False)
    status_output = gr.Textbox(label="✅ Status", interactive=False)

    process_button.click(process_video_ui, inputs=video_input, outputs=[summary_output, status_output])

    gr.Markdown("### ❓ Ask Questions About the Video")
    with gr.Row():
        question_input = gr.Textbox(label="📝 Ask a Question")
        qa_button = gr.Button("🤖 Get Answer")
    answer_output = gr.Textbox(label="💡 Answer", interactive=False)
    qa_button.click(ask_question_ui, inputs=question_input, outputs=answer_output)

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, gr_interface, path="/")

# FastAPI root route
@app.get("/")
def read_root():
    return {"message": "YouTube Summarizer is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT env variable
    uvicorn.run("app:app", host="0.0.0.0", port=port)

