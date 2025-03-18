import os
import cv2
import whisper
import yt_dlp
import requests
import json
import streamlit as st
from moviepy.editor import VideoFileClip
import asyncio

# Set Streamlit Page Config
st.set_page_config(page_title="AskVideo - AI Video Intelligence", page_icon="film_projector", layout="wide")

# Azure OpenAI Credentials
RESOURCE_NAME = "oai-use-promanager-dev"
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2024-02-15-preview"
API_KEY = "c54567fe809b451fac1a1b16eff22e0f"

# Azure OpenAI API URL
API_URL = f"https://{RESOURCE_NAME}.openai.azure.com/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
HEADERS = {"Content-Type": "application/json", "api-key": API_KEY}

# Function to download YouTube video
def download_youtube_video(url, output_path="video.mp4"):
    print("Downloading video...")
    if os.path.exists(output_path):
        os.remove(output_path)  # Ensure old video is deleted
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Video downloaded successfully!")
    return output_path

# Function to extract audio from video
def extract_audio(video_path, audio_path="audio.wav"):
    print("Extracting audio...")
    if os.path.exists(audio_path):
        os.remove(audio_path)  # Ensure the old file is deleted
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    print("Audio extracted successfully!")
    return audio_path

# Function to transcribe audio using Whisper (with timestamps)
def transcribe_audio_whisper(audio_path):
    print(f"Transcribing new audio file: {audio_path}")
    model = whisper.load_model("small", download_root="./models")  # Ensure fresh model load
    result = model.transcribe(audio_path, word_timestamps=True)  # Ensure word-level timestamps
    print("Transcription completed!")
    # Add timestamps in the transcript (Format: [timestamp] Text)
    transcript_with_timestamps = ""
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        transcript_with_timestamps += f"[{start:.2f}s - {end:.2f}s] {text}\n"
    return transcript_with_timestamps, result["segments"]

# Function to search inside past meeting transcripts with timestamps
def search_meeting_history(transcript, query, segments):
    search_results = []
    for segment in segments:
        if query.lower() in segment["text"].lower():
            search_results.append({
                "timestamp": f"{segment['start']:.2f}s - {segment['end']:.2f}s",
                "text": segment["text"]
            })
    
    if search_results:
        return search_results
    else:
        return ["No results found for your query."]

# Function to answer user questions using OpenAI (no timestamps in answer)
def answer_query(question, context, segments):
    system_prompt = "Answer the user's question based on the provided video transcript."
    query_data = {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_data), timeout=600)
    response_data = response.json()
    return response_data["choices"][0]["message"]["content"].strip() if "choices" in response_data else "Error: Failed to generate response."

# Streamlit UI
st.title("AskVideo - AI-Video Intelligence")
st.markdown("### Search, Analyze and Understand Video")

st.sidebar.header("Enter YouTube Link")
youtube_url = st.sidebar.text_input("Paste YouTube Video URL:")
if st.sidebar.button("Process Video"):
    if youtube_url:
        st.sidebar.info("Downloading video...")
        video_path = download_youtube_video(youtube_url)
        st.sidebar.success("Video Downloaded!")
        st.sidebar.info("Extracting audio...")
        audio_path = extract_audio(video_path)
        st.sidebar.success("Audio Extracted!")
        st.sidebar.info("Transcribing audio...")
        transcript, segments = transcribe_audio_whisper(audio_path)
        st.sidebar.success("Transcription Completed!")
        st.session_state["transcript"] = transcript
        st.session_state["segments"] = segments
        st.session_state["video_path"] = video_path

if "transcript" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Preview")
        st.video(st.session_state["video_path"])
    with col2:
        st.subheader("Video Transcript")
        st.text_area("Transcript:", st.session_state["transcript"], height=250)

    st.subheader("Search Inside Video")
    search_query = st.text_input("Enter a topic or keyword to search in the transcript:")
    if st.button("Search Records"):
        # Pass the segments to the search function
        search_results = search_meeting_history(st.session_state["transcript"], search_query, st.session_state["segments"])
        st.success("Search Results:")
        for result in search_results:
            if isinstance(result, dict):
                st.write(f"**Timestamp**: {result['timestamp']}")
                st.write(f"**Text**: {result['text']}")
            else:
                st.write(result)  # No results found case

    st.subheader("Ask Questions About the Video")
    user_query = st.text_input("Ask a question about the video:")
    if st.button("Ask"):
        answer = answer_query(user_query, st.session_state["transcript"], st.session_state["segments"])
        st.success("Answer:")
        st.write(answer)
