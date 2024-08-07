import streamlit as st
import time
import asyncio
import openai
from io import BytesIO
import keys
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from streamlit.runtime.scriptrunner import add_script_run_ctx

import queue
import threading
import tempfile
import os
import speech_recognition as sr
import numpy as np
from pydub import AudioSegment

import ffmpeg

st.set_page_config(layout="wide")

print("HI")
# Function to summarize text using OpenAI API
def summarize_text_with_openai_api(text, pro):
    client = openai.OpenAI(
        api_key=keys.api_key,
        base_url="https://api.ai71.ai/v1/"
    )
    prompts = {"summary": "Summarize the text in less words", "questions": "generate 7 questions based on this text to test my knowledge"}
    prompt = prompts[pro]
    
    try:
        response = client.chat.completions.create(
            model="tiiuae/falcon-180b-chat",
            messages= [{"role" :  "user", "content": f"{prompt}:\n\n{text}"}]
        )
        summary = response.choices[0].message.content[:-6]
        return summary
    except Exception as e:
        return f"Error: {e}"

# Initialize the recognizer
recognizer = sr.Recognizer()

# Streamlit UI Setup
if 'transcriptions' not in st.session_state:
    st.session_state['transcriptions'] = []
    st.session_state['action'] = None

# Streamlit UI Setup
# ... (keep the CSS styles as they were) ...

st.title('StudyScribe')

# Split the layout into two columns
left_col, middle_col, right_col = st.columns([1, 0.05, 2])

# Right Column Layout
notes_placeholder = st.empty()
with middle_col:
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

# Left Column Layout for actions
with left_col:
    if st.session_state['action'] == 'notes':
        st.header("Summarized Notes:")
        notes_placeholder = st.empty()
        if st.session_state['transcriptions']:
            text = '.'.join(st.session_state['transcriptions'])
            summarized_notes = summarize_text_with_openai_api(text, pro="summary")
            notes_placeholder.markdown(summarized_notes)
        else:
            notes_placeholder.markdown("No transcription to summarize")
    elif st.session_state['action'] == 'practice':
        st.header("Practice Questions:")
        notes_placeholder = st.empty()
        if st.session_state['transcriptions']:
            text = ''.join(st.session_state['transcriptions'])
            summarized_notes = summarize_text_with_openai_api(text, pro="questions")
            notes_placeholder.markdown(summarized_notes)
        else:
            notes_placeholder.markdown("No transcription to summarize")

with right_col:
    # WebRTC setup
    audio_buffer = queue.Queue()

    def audio_frame_callback(frame):
        audio_buffer.put(frame.to_ndarray().tobytes())

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
    )

    transcription_queue = queue.Queue()

    def transcribe_audio():
        while True:
            if webrtc_ctx.state.playing:
                frames = []
                silent_counter = 0
                while silent_counter < 10:  # Collect about 1 second of audio
                    try:
                        frame = audio_buffer.get(timeout=0.1)
                        frames.append(frame)
                        frame_avg = 0
                        for i in frame:
                            frame_avg += i
                        frame_avg /= len(frame)
                        print(frame_avg)
                        if frame_avg < 90:
                            print('here')
                            silent_counter += 1
                        else:
                            silent_counter = 0
                    except queue.Empty:
                        break

                if len(frames) > 0:
                    audio_data = []
                    for frame in frames:
                        audio_data += frame
                    audio_segment = AudioSegment(
                        bytes(audio_data),
                        frame_rate=44100,
                        sample_width=4,
                        channels=1
                    )

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                        audio_segment.set_frame_rate(16000)
                        audio_segment.export(temp_audio_file.name, format="wav")
                        
                        with sr.AudioFile(temp_audio_file.name) as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio_content = recognizer.record(source)
                            try:
                                transcription = recognizer.recognize_google(audio_content)
                                if transcription:
                                    transcription_queue.put(transcription)
                            except sr.UnknownValueError:
                                pass
                            except sr.RequestError as e:
                                st.error(f"Could not request results; {e}")
                    
                    os.unlink(temp_audio_file.name)
            else:
                break

    # Start transcription in a separate thread
    thread = threading.Thread(target=transcribe_audio, daemon=True).start()
    add_script_run_ctx(thread)

    def update_transcriptions():
        while not transcription_queue.empty():
            transcription = transcription_queue.get()
            st.session_state['transcriptions'].append(transcription)
        # Update the transcription area
        trp.markdown(r"<div class='scrollable-container'>" + "<br>".join(f"{i+1}: {t}" for i, t in enumerate(st.session_state['transcriptions'])) + r"<style> .scrollable-container {height: 400px; overflow-y: auto;}</style> </div>", unsafe_allow_html=True)

    # Transcription area with a scrollbar
    trp = st.markdown(r"<div class='scrollable-container'>" + "<br>".join(f"{i+1}: {t}" for i, t in enumerate(st.session_state['transcriptions'])) + r"<style> .scrollable-container {height: 400px; overflow-y: auto;}</style> </div>", unsafe_allow_html=True)

    # Buttons in the bottom 1/3 of the right column
    with st.container():
        get_notes, _, get_practice = st.columns(3)
        if get_notes.button('Get Notes', key='1'):
            st.session_state['action'] = 'notes'

        if get_practice.button('Get Practice Questions', key='3'):
            st.session_state['action'] = 'practice'

    # File uploader for audio files
    uploaded_file = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        audio = sr.AudioFile(BytesIO(audio_data))
        with audio as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_content = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_content)
                st.session_state['transcriptions'].append(transcription)
                st.success("File transcribed successfully!")
            except sr.UnknownValueError:
                st.error("Could not understand the audio") 
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")

# Continuously update the transcriptions in the main thread
while True:
    update_transcriptions()
    time.sleep(1)
