import streamlit as st
from vidcrawl import separator, analyse_images, transcribe_audio_s3
from vidcrawl._merger import create_unified_report, save_report
import tempfile
import os

st.set_page_config(page_title="Video Q&A", layout="centered")
st.title("🎥 Video RAG Service")

# Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Optional: Ask question input
user_question = st.text_input("Ask a question about the video:")

# Submit button
if st.button("Process Video"):
    if uploaded_video is None:
        st.warning("Please upload a video first.")
    else:
        with st.spinner("Processing video... this might take a while ⏳"):
            try:
                # Save uploaded video to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    tmp_path = tmp_file.name

                # --- Call your existing functions directly ---
                audio, video = separator(tmp_path)
                visual_analysis = analyse_images(video)
                transcripts = transcribe_audio_s3(audio)
                report_str = create_unified_report(visual_analysis, transcripts, video, audio)

                save_report(report_str)

                # Display results in Streamlit
                st.success("✅ Video processed successfully!")
                st.video(uploaded_video)
                st.subheader("Generated Report:")
                st.write(report_str)

            except Exception as e:
                st.error(f"Error during processing: {e}")

            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
