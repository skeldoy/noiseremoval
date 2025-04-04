import sounddevice as sd
from scipy.io.wavfile import write, read
from faster_whisper import WhisperModel
import ollama
from TTS.api import TTS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize components
whisper_model = WhisperModel("large-v3")
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_DIM = 384  # Dimensions for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(EMBEDDING_DIM)

import re

def remove_emojis(text):
    return re.sub(r'[\U0001F600-\U0001F64F\\U0001F300-\U0001F5FF\\U0001F680-\U0001F6FF\\U0001F700-\U0001F77F\\U0001F780-\U0001F7FF\\U0001F800-\U0001F8FF\\U0001F900-\U0001F9FF\\U0001FA00-\U0001FA6F\\U0001FA70-\U0001FAFF\\U0001FB00-\U0001FBFF\\U0001FC00-\U0001FCFF\\U0001FD00-\U0001FDFF\\U0001FE00-\U0001FEFF\\U0001FF00-\U0001FFFF]', '', text)

def get_embedding(text):
    return embedder.encode(text)

def add_to_faiss(text):
    embedding = np.array([get_embedding(text)], dtype=np.float32)
    index.add(embedding)

def retrieve_context(query_text, top_k=3):
    query_embedding = np.array([get_embedding(query_text)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return indices  # In a full system, store texts alongside embeddings

def record_audio(duration=5):
    sample_rate = 44100
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    write("input.wav", sample_rate, recording)

def transcribe_audio():
    segments, _ = whisper_model.transcribe("input.wav")
    return " ".join([segment.text for segment in segments])

def generate_response(prompt):
    response = ollama.generate(model="gemma2", prompt=prompt)
    return response["response"]


def synthesize_speech(text, output_file="hal_output.wav"):
    cleaned_text = remove_emojis(text)  # Remove emojis
    tts.tts_to_file(text=cleaned_text, file_path=output_file)  # Correct method for Coqui TTS


def play_audio(file):
    sample_rate, audio = read(file)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

# Main loop
while True:
    print("Listening...")
#    record_audio()
#    user_text = transcribe_audio()
    user_text = "Hello my dear friend. How are you today?"
    print(f"User: {user_text}")
    
    # Retrieve context from FAISS
    context_indices = retrieve_context(user_text)
    # In a full system, fetch stored texts using indices and append to prompt
    
    hal_response = generate_response(user_text)  # Simplified; include context in real implementation
    print(f"HAL: {hal_response}")
    
    synthesize_speech(hal_response)
    play_audio("hal_output.wav")
    
    # Store response in FAISS
    add_to_faiss(f"User: {user_text} | HAL: {hal_response}")

