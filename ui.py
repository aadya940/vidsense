import streamlit as st
# Assuming these imports are correctly pointing to your project's modules
from vidcrawl import separator, analyse_images, transcribe_audio_s3
from vidcrawl._merger import create_unified_report, save_report
import tempfile
import os
import boto3 # Ensure boto3 is imported
import json # Ensure json is imported
import traceback # Ensure traceback is imported

st.set_page_config(page_title="Video Q&A", layout="centered")
st.title("🎥 Video RAG Service")

# Initialize session state for report path and processing status
if 'report_path' not in st.session_state:
    st.session_state.report_path = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

# --- Video Upload ---
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# --- Video Processing Section ---
if st.button("Process Video"):
    if uploaded_video is None:
        st.warning("⚠️ Please upload a video first.")
    else:
        # Define tmp_path outside the try block for reliable cleanup
        tmp_path = None
        # Define and create a specific temp directory for this app
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_temp_dir = os.path.join(script_dir, "streamlit_temp_files")
        os.makedirs(app_temp_dir, exist_ok=True)

        with st.spinner("Processing video... this might take a while ⏳"):
            try:
                # Save uploaded video to the defined temp directory
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=app_temp_dir) as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    tmp_path = tmp_file.name # Get the full path


                # --- Call your backend processing functions ---
                # Assuming separator returns path(s) or relevant data structures
                audio_path, video_frames_or_clips = separator(tmp_path)
                # Assuming analyse_images uses the video output from separator
                visual_analysis = analyse_images(video_frames_or_clips)
                # Assuming transcribe_audio_s3 handles audio path (might need S3 upload first if local)
                transcripts = transcribe_audio_s3(audio_path)
                # Assuming create_unified_report combines the results
                report_str = create_unified_report(visual_analysis, transcripts, video_frames_or_clips, audio_path)

                # Save the generated report and store its path
                # Assuming save_report saves to a file and returns the path
                report_file_path = save_report(report_str)
                st.session_state.report_path = report_file_path
                st.session_state.video_processed = True # Mark processing as complete

                st.success("✅ Video processed successfully!")
                st.info(f"📄 Report saved to: {report_file_path}")

            except Exception as e:
                st.error(f"Error during video processing: {e}")
                st.code(traceback.format_exc()) # Display detailed error traceback

            finally:
                # --- Cleanup temporary files ---
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                        st.info(f"Cleaned up temporary video file: {tmp_path}")
                    except Exception as cleanup_e:
                        st.warning(f"Could not remove temporary video file {tmp_path}: {cleanup_e}")
                # Add cleanup for audio_path if it's a local temp file created by separator
                if 'audio_path' in locals() and isinstance(audio_path, str) and os.path.exists(audio_path) and app_temp_dir in audio_path:
                     try:
                        os.remove(audio_path)
                        st.info(f"Cleaned up temporary audio file: {audio_path}")
                     except Exception as cleanup_e:
                        st.warning(f"Could not remove temporary audio file {audio_path}: {cleanup_e}")

# --- Question Answering Section ---
# Only show this section after a video has been processed successfully
if st.session_state.video_processed:
    st.divider() # Visual separator
    st.subheader("💬 Ask Questions About Your Video")

    # Input field for the user's question
    user_question = st.text_input("What would you like to know about the video?",
                                  placeholder="e.g., What are the main topics discussed?",
                                  key="question_input") # Key helps maintain input value

    # "Ask" button to trigger the analysis
    if st.button("Ask", type="primary"):
        if not user_question:
            st.warning("⚠️ Please enter a question.")
        elif not st.session_state.report_path or not os.path.exists(st.session_state.report_path):
             st.error("❌ Report file is missing. Please process the video again.")
        else:
            report_content = None # Initialize variable to hold report text
            try:
                # Attempt to read the previously generated report file
                with open(st.session_state.report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            except Exception as e:
                st.error(f"Error reading report file ({st.session_state.report_path}): {e}")

            # Proceed only if the report content was successfully read
            if report_content:
                with st.spinner("Thinking... 🤔"):
                    try:
                        # Initialize AWS Bedrock client - relies on correctly configured AWS credentials
                        # (Environment variables, ~/.aws/credentials, or IAM role)
                        bedrock_runtime = boto3.client(
                            service_name='bedrock-runtime',
                            region_name=os.getenv('AWS_REGION', 'us-east-1')
                        )

                        # Construct the prompt for the LLM, including the report and the question
                        # For Converse API, we structure this into system prompt and user message
                        system_prompt = f"""You are a helpful assistant analyzing video content.
Below is a comprehensive report about a video including visual analysis and transcripts.
Answer the user's question based ONLY on the video report provided. Be specific and concise.

VIDEO REPORT:
{report_content}
"""
                        user_message = {"role": "user", "content": [{"text": user_question}]}

                        target_model_id = 'amazon.nova-pro-v1:0'
                        

                        # --- Call the Bedrock Converse API ---
                        response = bedrock_runtime.converse(
                            modelId=target_model_id,
                            messages=[user_message], # Pass the user's question
                            system=[{"text": system_prompt}], # Pass the report as system context
                            inferenceConfig={ # Inference parameters for Converse API
                                "maxTokens": 2000,
                                "temperature": 0.7
                                # Add stopSequences if needed: "stopSequences": ["\n\nHuman:"]
                            }
                            # Additional config like toolConfig could be added here if needed
                        )
                        # ------------------------------------

                        # --- Parse the response from the Converse API ---
                        # The answer is typically in response['output']['message']['content'][0]['text']
                        if (response and 'output' in response and
                            'message' in response['output'] and
                            'content' in response['output']['message'] and
                            len(response['output']['message']['content']) > 0 and
                            'text' in response['output']['message']['content'][0]):

                            answer = response['output']['message']['content'][0]['text'].strip()
                        else:
                            # Handle cases where the expected output structure is missing
                            st.error("❌ Could not parse the response from the Converse API.")
                            st.json(response) # Show the raw response for debugging
                            raise ValueError("Unexpected response format from Converse API.")
                        # ----------------------------------------------

                        # Display the generated answer
                        st.success("**Answer:**")
                        st.markdown(answer)

                    except Exception as e:
                        # Catch errors during the Bedrock call or response parsing
                        st.error(f"Error querying the AI model: {e}")
                        st.info("💡 Please verify your AWS credentials, Bedrock model access, and region configuration.")
                        st.code(traceback.format_exc()) # Show detailed traceback


    # --- Download Report Button ---
    # Show button only if the report path exists
    if st.session_state.report_path and os.path.exists(st.session_state.report_path):
        try:
            # Read the report content again for download
            with open(st.session_state.report_path, 'r', encoding='utf-8') as f:
                report_data = f.read()
            st.download_button(
                label="📥 Download Full Report",
                data=report_data,
                file_name="video_analysis_report.md", # Suggested filename
                mime="text/markdown"                   # Correct MIME type
            )
        except Exception as e:
            st.warning(f"Could not read report file for download: {e}")

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1.  **Upload** your video (MP4, MOV, AVI).
    2.  Click **Process Video** and wait.
    3.  Once processed, **Ask Questions** about it!
    """)
    st.divider()
    st.header("❓ Example Questions")
    st.markdown("""
    - "Summarize the video."
    - "What are the key topics discussed?"
    - "List the objects visible in the first scene."
    - "Is the setting indoors or outdoors?"
    """)

    # Button to reset the state and allow processing a new video
    if st.session_state.video_processed:
        st.divider()
        if st.button("🔄 Process Another Video"):
            # Clear relevant session state variables
            st.session_state.report_path = None
            st.session_state.video_processed = False
            # Rerun the script to reset the UI
            st.rerun()