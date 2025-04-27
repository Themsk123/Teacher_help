from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import re, os
from dotenv import load_dotenv

load_dotenv() 
api_key=os.getenv('GROQ_API_KEY')


# STEP 1: Improved YouTube transcript handling
def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",        # Standard URL
        r"youtu\.be/([a-zA-Z0-9_-]{11})", # Short URL
        r"embed/([a-zA-Z0-9_-]{11})",     # Embed URL
        r"^([a-zA-Z0-9_-]{11})$"          # Direct ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL")

def get_transcript(video_url):
    try:
        video_id = extract_video_id(video_url)
        transcript_data = None  # Initialize to avoid UnboundLocalError
        
        # Try English first, then any available language
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except NoTranscriptFound:
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Find first available transcript (including auto-generated)
            for transcript in transcript_list:
                try:
                    transcript_data = transcript.fetch()
                    break  # Exit loop once a valid transcript is found
                except Exception as e:
                    print(f"Skipping transcript {transcript.language_code}: {str(e)}")
                    continue

            # If no transcript was found after checking all
            if transcript_data is None:
                raise NoTranscriptFound(video_id, ('en',), [])

        return " ".join([t["text"].strip() for t in transcript_data])
    
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video")
    except Exception as e:
        raise ValueError(f"Error fetching transcript: {str(e)}")




# STEP 2: Create prompt content based on user preferences
def get_prompt_content(transcript, num_pages, user_input=""):
    return f"""
 You are a helpful note-making assistant. Summarize the following YouTube transcript into well-structured, concise notes that fit approximately {num_pages} A4 pages or slides. The content should be educational, formatted with bullet points, subheadings, and concise explanations.

    For each page or slide:
    - Provide a **clear, informative title** for the section (e.g., "Introduction to AI", "Machine Learning Basics").
    - Use **bullet points** to highlight the main points and key takeaways.
    - Include **examples** where applicable, making them short and easy to understand.
    - Break down complex topics with **subheadings** for clarity.
    - Ensure that each section is easy to read, concise, and directly relevant to the transcript.

    The notes should be divided as follows:
    - **Pages/Slides should be formatted** with minimal but effective text that fits neatly into an educational slide or note format.
    - **Avoid long paragraphs**, focus on making the content **digestible** for students or teachers.

    Keep the tone educational and accessible to a wide range of learners.

    **Important:**
    - If the user asks for anything **unrelated** or **irrelevant** to summarizing the transcript or making the notes (e.g., personal requests, off-topic queries), **do not answer**.
    - Strictly focus only on the task at hand: summarizing the YouTube transcript into well-structured, concise notes.
    - If a query is irrelevant, respond with: "This is beyond the scope of the current task. Please provide relevant instructions related to the note-making process."
    

Transcript:
{transcript}
"""

# STEP 3: Memory Setup
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# STEP 4: Setup Model
llm=ChatGroq(model='gemma2-9b-it',api_key=api_key)
model_with_memory = RunnableWithMessageHistory(llm, get_session_history)

# STEP 5: Main entry function
def run_note_maker(video_url, num_pages, session_id="firstchat", feedbacks=[]):
    config = {"configurable": {"session_id": session_id}}

    transcript = get_transcript(video_url)
    initial_prompt = get_prompt_content(transcript, num_pages)
    
    # Initial Summary
    response = model_with_memory.invoke([HumanMessage(content=initial_prompt)], config=config)
    results = [("\n--- Initial Summary ---\n", response.content)]

    # Feedbacks
    for msg in feedbacks:
        reply = model_with_memory.invoke([HumanMessage(content=msg)], config=config)
        results.append((f"\n--- Feedback: {msg} ---\n", reply.content))
    
    return results
