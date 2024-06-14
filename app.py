import streamlit as st
import os
import tempfile
import shutil
import ffmpeg
import subprocess
import openai
from pytube import YouTube

CUSTOM_INSTRUCTIONS = "Summarize the video in a nice markdown format."

def download_youtube_video(youtube_url):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    # Create a temporary file name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        temp_file_name = tmp_file.name
    
    # Check if the file already exists and delete it if necessary
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
    
    # Download the audio stream to the temporary file
    audio_stream.download(output_path=os.path.dirname(temp_file_name), filename=os.path.basename(temp_file_name))
    
    return temp_file_name

def save_video_to_disk(video_file):
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + video_file.name.split('.')[-1]) as tmp_file:
            shutil.copyfileobj(video_file, tmp_file)
            return tmp_file.name
    return None

def format_video(video_file, is_youtube=False):
    video_path = save_video_to_disk(video_file) if not is_youtube else video_file
    print(video_path)
    print(is_youtube)
    if video_path is not None:
        output_audio_path = video_path.rsplit('.', 1)[0] + '.wav'
        ffmpeg.input(video_path).output(output_audio_path, acodec='pcm_s16le', ar=16000, ac=2).run()
        return output_audio_path

def transcribe_audio(audio_file):
    output_file = audio_file.rsplit('.', 1)[0]
    command = f'/home/ubuntu/tools/whisper.cpp/main -m /home/ubuntu/tools/whisper.cpp/models/ggml-distil-large-v3.bin -f "{audio_file}" --output-txt --output-file "{output_file}"'
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode == 0:
        with open(output_file + '.txt', 'r') as file:
            return file.read()
    else:
        raise Exception(f"Error in transcription: {process.stderr}")
    
def query_transcription(transcription, openai_api_key, instructions):
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": transcription}]
    )
    return response.choices[0].message.content

if "output" not in st.session_state:
    st.session_state["output"] = ""

st.title("🎥 Chat with Videos")

with st.form("upload_form"):
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "webm", "ts"])
    youtube_url = st.text_input("Youtube URL", placeholder="https://www.youtube.com/watch?v=...")
    custom_instructions = st.text_area("Query", value=CUSTOM_INSTRUCTIONS)
    openai_api_key = st.text_input("OpenAI API Key", placeholder="sk-...")
    submitted = st.form_submit_button("Submit")

validation_placeholder = st.empty()

if submitted:
    if video_file is None and youtube_url is None:
        validation_placeholder.error("Please upload a video or enter a Youtube URL")
    if openai_api_key is None:
        validation_placeholder.error("Please enter an OpenAI API key")

    with st.status("Processing..."):
        try:
            if youtube_url:
                st.write("Downloading video...")
                video_file = download_youtube_video(youtube_url)
            st.write("Converting video format...")
            print(youtube_url)
            audio_file = format_video(video_file, is_youtube=youtube_url != "")
            st.write("Transcribing audio...")
            transcription = transcribe_audio(audio_file)
            st.write("Querying...")
            output = query_transcription(transcription, openai_api_key, custom_instructions)
            st.session_state["output"] = output
            st.success("Done!")
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state["output"]:
    st.text_area("Output", st.session_state["output"])