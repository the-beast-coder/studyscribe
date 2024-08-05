import streamlit as st
import asyncio
import openai
import speech_recognition as sr
import keys


st.set_page_config(layout="wide")

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
    st.session_state['run'] = False
    st.session_state['action'] = None

title_alignment="""
<style>
.st-emotion-cache-10trblm {
  text-align: center !important;
  margin-left: 0;
}
</style> """

vertical_line_css = """
<style>
.vertical-line {
    border-left: 2px solid #000;
    height: 500px;
}
</style>
"""

st.markdown(vertical_line_css, unsafe_allow_html=True)
st.markdown(title_alignment, unsafe_allow_html=True)
st.title('StudyScribe')

# Split the layout into two columns
left_col, middle_col, right_col = st.columns([1, 0.05, 2])

# Right Column Layout
notes_placeholder = st.empty()
with middle_col:
    st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)  # This column will display as a vertical line

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
    start, stop = st.columns(2)
    start.button('Start listening', on_click=lambda: start_listening(), use_container_width=True)
    stop.button('Stop listening', on_click=lambda: stop_listening(), use_container_width=True)

    # Transcription area with a scrollbar
    trp = st.markdown(r"<div class='scrollable-container'>" + "<br>".join(f"{i+1}: {t}" for i, t in enumerate(st.session_state['transcriptions'])) + r"<style> .scrollable-container {height: 400px; overflow-y: auto;}</style> </div>", unsafe_allow_html=True)
    # Placeholder for displaying the current partial or final transcription
    current_transcription_placeholder = st.empty()

    # Buttons in the bottom 1/3 of the right column
    with st.container():
        get_notes, _, get_practice = st.columns(3)
        if get_notes.button('Get Notes', key='1'):
            st.session_state['action'] = 'notes'
            # Get the last transcription
            if st.session_state['transcriptions']:
                last_transcription = st.session_state['transcriptions'][-1]

        if get_practice.button('Get Practice Questions', key='3'):
            st.session_state['action'] = 'practice'

# Functions to handle start and stop listening
def start_listening():
    st.session_state['run'] = True
    asyncio.run(transcribe_audio())

def stop_listening():
    st.session_state['run'] = False

async def transcribe_audio():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while st.session_state['run']:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                try:
                    transcription = recognizer.recognize_google(audio)
                    st.session_state['transcriptions'].append(transcription)
                    # Update the transcription area within the scrollable container
                    trp.markdown(r"<div class='scrollable-container'>" + "<br>".join(f"{i+1}: {t}" for i, t in enumerate(st.session_state['transcriptions'])) + r"<style> .scrollable-container {height: 400px; overflow-y: auto;}</style> </div>", unsafe_allow_html=True)
                    current_transcription_placeholder.markdown("**Listening for more...**")
                except sr.UnknownValueError or sr.RequestError:
                    pass
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                pass