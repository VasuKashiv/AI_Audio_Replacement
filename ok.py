import streamlit as st
import requests
from moviepy.editor import VideoFileClip, AudioFileClip
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent
import os
import subprocess

GOOGLE_APPLICATION_CREDENTIALS = "path/to/your/credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

st.title("AI-Powered Audio Replacement for Video")
st.write("Upload a video file to replace its audio with AI-generated voice.")

azure_openai_key = "YOUR_AZURE_OPENAI_KEY"  
azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

def convert_audio(input_path, output_path):
    try:
        # Run the ffmpeg command
        command = [
            "ffmpeg",
            "-i", input_path,      # Input file
            "-ar", "16000",        # Set the sample rate to 16000 Hz
            "-ac", "1",            # Set the audio to mono
            output_path            # Output file
        ]
        subprocess.run(command, check=True)
        print(f"Audio converted successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {str(e)}")


def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path

def transcribe_audio(audio_path):
    client = speech.SpeechClient()
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )
    response = client.recognize(config=config, audio=audio)
    words_with_timestamps = []

    for result in response.results:
        for word_info in result.alternatives[0].words:
            words_with_timestamps.append({
                "word": word_info.word,
                "start_time": word_info.start_time.total_seconds(),  # Start time 
                "end_time": word_info.end_time.total_seconds()       # End time 
            })
    
    return words_with_timestamps

def correct_transcription_with_azure_openai(transcription):
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key
    }
    # Data to be sent to Azure OpenAI
    data = {
        "messages": [{"role": "user", "content": f"Correct the following transcription, removing grammatical mistakes and filler words: {transcription}"}],
        "max_tokens": 1000
    }
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        corrected_text = result["choices"][0]["message"]["content"].strip()
        return corrected_text
    else:
        st.error(f"Failed to get response from Azure OpenAI: {response.status_code} - {response.text}")
        return None

def align_timestamps(original_words, corrected_text):
    """Align corrected words with the original timestamps."""
    corrected_words = corrected_text.split()  
    
    aligned_timestamps = []
    original_index = 0
    
    for corrected_word in corrected_words:
        # find the closest matching original word based on its position
        if original_index < len(original_words):
            original_word_info = original_words[original_index]
            aligned_timestamps.append({
                "word": corrected_word,
                "start_time": original_word_info["start_time"],
                "end_time": original_word_info["end_time"]
            })
            original_index += 1 

    return aligned_timestamps

def synthesize_speech_phrase(text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        name="en-US-Wavenet-J", 
        language_code="en-US"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    output_audio_path = "generated_audio.wav"
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
    return AudioSegment.from_wav(output_audio_path)

def detect_non_voice_segments(audio_segment, silence_thresh=-50, min_silence_len=1000):
    """Detects silence in the given audio segment."""
    non_voice_segments = detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    non_voice_segments += detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh + 10)
    
    return sorted(non_voice_segments)

def suppress_original_voice(audio_segment, low_freq=300, high_freq=3400):
    """
    Suppress the original voice frequencies while retaining background music or noise.
    """
    filtered_audio = audio_segment.low_pass_filter(high_freq).high_pass_filter(low_freq)
    return filtered_audio

def create_corrected_audio(aligned_timestamps, original_audio_path, original_duration):
    original_audio = AudioSegment.from_wav(original_audio_path)
    # Step 1: Suppress the original voice frequencies in the audio
    music_and_sound_only_audio = suppress_original_voice(original_audio)
    # Step 2: Detect low-energy non-voice segments (including silence)
    non_voice_segments = detect_non_voice_segments(original_audio)
    
    corrected_audio = AudioSegment.silent(duration=0) 
    
    # Process phrases at the sentence level
    phrase_start_time = None
    current_phrase = []

    # Step 3: Iterate through the aligned timestamps and synthesize speech
    for i, timestamp in enumerate(aligned_timestamps):
        word_audio = synthesize_speech_phrase(timestamp["word"])
        
        if phrase_start_time is None:
            phrase_start_time = timestamp["start_time"]

        current_phrase.append(word_audio)

        # If end of phrase or end of timestamps reached
        if i == len(aligned_timestamps) - 1 or \
           aligned_timestamps[i + 1]["start_time"] - timestamp["end_time"] > 0.5:
            phrase_audio = sum(current_phrase)
            phrase_duration = len(phrase_audio)
            original_phrase_duration = (timestamp["end_time"] - phrase_start_time) * 1000  
            
            # Add silence or background noise to match timing
            if phrase_duration < original_phrase_duration:
                silence_duration = original_phrase_duration - phrase_duration
                phrase_audio += AudioSegment.silent(duration=silence_duration)
            
            corrected_audio += phrase_audio
            phrase_start_time = None
            current_phrase = []

    # Step 4: Now adjust based on detected non-voice segments in the original audio
    for segment in non_voice_segments:
        start, end = segment
        # Insert the non-voice sound from the music-only track at the correct position
        corrected_audio = corrected_audio[:start] + music_and_sound_only_audio[start:end] + corrected_audio[start:]

    # Step 5: Adjust the final audio duration to match the original video
    if len(corrected_audio) < original_duration:
        corrected_audio += AudioSegment.silent(duration=original_duration - len(corrected_audio))
    elif len(corrected_audio) > original_duration:
        corrected_audio = corrected_audio.speedup(playback_speed=len(corrected_audio) / original_duration)

    corrected_audio_path = "corrected_audio.wav"
    corrected_audio.export(corrected_audio_path, format="wav")
    return corrected_audio_path

def replace_audio_in_video(original_video_path, new_audio_path):
    video_clip = VideoFileClip(original_video_path)
    new_audio_clip = AudioFileClip(new_audio_path)
    final_video = video_clip.set_audio(new_audio_clip)
    final_output_path = "final_video_with_replaced_audio.mp4"
    final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    return final_output_path

if uploaded_video:
    st.info("Processing your video, please wait...")
    # video_path = r"C:\Users\Lenovo\Downloads\sample1.mp4"
    video_path = uploaded_video.name
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.video(video_path)

    audio_path = extract_audio(video_path)
    st.write("Audio extracted from the video.")
    convert_audio(audio_path,"final_audio_path.wav")
    transcription_map = transcribe_audio("final_audio_path.wav")
    original_transcription = " ".join([entry["word"] for entry in transcription_map])
    st.write("Original Transcription:", original_transcription)

    corrected_text = correct_transcription_with_azure_openai(original_transcription)
    if corrected_text:
        st.write("Corrected Transcription:", corrected_text)

        aligned_timestamps=align_timestamps(transcription_map,corrected_text)
        original_video_clip = VideoFileClip(video_path)
        original_video_duration = original_video_clip.duration * 1000  
        corrected_audio_path=create_corrected_audio(aligned_timestamps,"final_audio_path.wav", original_video_duration)
        st.write("Corrected audio generated.")

        final_video_path = replace_audio_in_video(video_path, corrected_audio_path)
        st.success("Audio replaced successfully! Here is the final video:")

        st.video(final_video_path)

         # Cleanup: Remove unnecessary files
        os.remove(audio_path) 
        os.remove("final_audio_path.wav")  
        os.remove(corrected_audio_path)  
        os.remove(video_path)
    else:
        st.error("Failed to get a corrected transcription.")
