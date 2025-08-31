import os
import re
import PyPDF2
import streamlit as st
from TTS.api import TTS
from pydub import AudioSegment
from io import BytesIO


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of PDF and clean it."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    text = re.sub(r"Scan to Download", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text, max_chars=1500):
    """Split long text into safe chunks for TTS."""
    chunks = []
    while len(text) > max_chars:
        split_index = text[:max_chars].rfind(".")
        if split_index == -1:
            split_index = max_chars
        chunk = text[:split_index + 1].strip()
        if len(chunk) > 20:  # avoid tiny fragments
            chunks.append(chunk)
        text = text[split_index + 1:]
    if len(text.strip()) > 20:
        chunks.append(text.strip())
    return chunks


def generate_book_tts(text, output_file):
    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

    speaker = tts.speakers[0]
    st.info(f"[+] Using speaker: {speaker}")

    # Split text into manageable chunks
    chunks = split_text(text)
    st.info(f"[+] Total chunks to process: {len(chunks)}")

    temp_files = []
    progress = st.progress(0)

    for i, chunk in enumerate(chunks):
        temp_file = f"chunk_{i}.wav"
        try:
            tts.tts_to_file(text=chunk, speaker=speaker, file_path=temp_file)
            temp_files.append(temp_file)
            progress.progress((i + 1) / len(chunks))
        except Exception as e:
            st.warning(f"[!] Skipping chunk {i} due to error: {e}")


    final_audio = AudioSegment.empty()
    for temp_file in temp_files:
        final_audio += AudioSegment.from_wav(temp_file)
        os.remove(temp_file)

    final_audio.export(output_file, format="wav")

    audio_bytes = BytesIO()
    final_audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    return audio_bytes



def convert_to_user_voice(reference_voice_file, input_speech, output_file, chunk_ms=15000):

    if hasattr(reference_voice_file, "read"):
        ref_path = os.path.join(OUTPUT_DIR, "user_voice_sample.wav")
        with open(ref_path, "wb") as f:
            f.write(reference_voice_file.read())
    else:
        ref_path = reference_voice_file  # already a path

    try:
        vc_tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", gpu=False)
        st.info("[+] Using voice conversion model: freevc24")
    except Exception as e:
        st.warning(f"[!] freevc24 not available, falling back to freevc. Error: {e}")
        vc_tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc", gpu=False)
        st.info("[+] Using voice conversion model: freevc")

    
    audio = AudioSegment.from_wav(input_speech)
    chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]

    converted_audio = AudioSegment.empty()

    for i, chunk in enumerate(chunks):
        temp_in = os.path.join(OUTPUT_DIR, f"temp_in_{i}.wav")
        temp_out = os.path.join(OUTPUT_DIR, f"temp_out_{i}.wav")

        chunk.export(temp_in, format="wav")

        try:
            vc_tts.voice_conversion_to_file(
                source_wav=temp_in,
                target_wav=ref_path,
                file_path=temp_out
            )
            converted_audio += AudioSegment.from_wav(temp_out)
        except Exception as e:
            st.warning(f"[!] Skipping chunk {i} due to error: {e}")

        
        if os.path.exists(temp_in):
            os.remove(temp_in)
        if os.path.exists(temp_out):
            os.remove(temp_out)

    # Save final merged audio
    converted_audio.export(output_file, format="wav")
    return output_file



def main():
    st.title("📚 PDF to Speech Converter with Voice Cloning")
    st.write("Upload a PDF and convert it into speech (audiobook). Then clone it into your voice!")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    voice_sample = st.file_uploader("Upload 15s Voice Sample (WAV)", type="wav")

    if uploaded_file:
        pdf_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        base_name = os.path.splitext(uploaded_file.name)[0]
        output_speech_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
        converted_speech = os.path.join(OUTPUT_DIR, f"{base_name}_converted.wav")

        # If speech already exists
        if os.path.exists(output_speech_path):
            st.success("✅ Speech already generated for this PDF.")
            with open(output_speech_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
            if st.button("Regenerate Speech"):
                text = extract_text_from_pdf(pdf_path)
                if text:
                    st.info("Generating speech...")
                    audio_bytes = generate_book_tts(text, output_speech_path)
                    st.success("🎉 Speech generation complete!")
                    st.audio(audio_bytes, format="audio/wav")
        else:
            if st.button("Generate Speech"):
                text = extract_text_from_pdf(pdf_path)
                if text:
                    st.info("Generating speech...")
                    audio_bytes = generate_book_tts(text, output_speech_path)
                    st.success("🎉 Speech generation complete!")
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.error("No text could be extracted from this PDF.")

        # Voice Conversion section
        if os.path.exists(output_speech_path) and voice_sample:
            st.write("### 🎤 Convert Speech to Your Voice")
            if os.path.exists(converted_speech):
                st.success("✅ Converted speech already exists.")
                st.audio(converted_speech, format="audio/wav")
                if st.button("Regenerate Converted Speech"):
                    output = convert_to_user_voice(voice_sample, output_speech_path, converted_speech)
                    st.success("🎉 Conversion complete!")
                    st.audio(output, format="audio/wav")
            else:
                if st.button("Convert to My Voice"):
                    output = convert_to_user_voice(voice_sample, output_speech_path, converted_speech)
                    st.success("🎉 Conversion complete!")
                    st.audio(output, format="audio/wav")


if __name__ == "__main__":
    main()