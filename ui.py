import streamlit as st
import os
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import boto3
from dotenv import load_dotenv

from vidcrawl import separator, analyse_images, transcribe_audio_s3
from vidcrawl._merger import create_unified_report
from vidcrawl.core.datamodel import Transcript, AudioClip, VideoClip

load_dotenv()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Chat with Your Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: white;
        color: #333;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .message-sender {
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 12px;
        opacity: 0.8;
    }
    
    /* Status badges */
    .status-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    
    .status-success {
        background: #4CAF50;
        color: white;
    }
    
    .status-processing {
        background: #FF9800;
        color: white;
    }
    
    .status-waiting {
        background: #2196F3;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 12px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    /* Headers */
    h1 {
        color: #667eea;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    /* Upload section */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.2);
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bedrock_client():
    """Create Bedrock client for chat"""
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

def analyze_video_parallel(
    video_path: str,
    num_frames: int = 30,
    sample_rate: int = 20
) -> Tuple[str, List[Transcript], AudioClip, VideoClip]:
    """Optimized pipeline: Run transcription and visual analysis in parallel."""    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        audio_clip, video_clip = separator(video_path, num_frames, sample_rate)
        
        visual_future = executor.submit(analyse_images, video_clip)
        audio_future = executor.submit(transcribe_audio_s3, audio_clip)
        
        visual_analysis = visual_future.result()
        transcripts = audio_future.result()
    
    return visual_analysis, transcripts, audio_clip, video_clip


def chat_with_video(
    user_message: str,
    video_report: str,
    chat_history: List[Dict],
    model_id: str = "us.amazon.nova-pro-v1:0"
) -> str:
    """Chat with the video analysis using Bedrock.
    
    Args:
        user_message: The user's message/query
        video_report: Pre-generated text analysis of the video
        chat_history: List of previous messages in the conversation
        model_id: ID of the Bedrock model to use (default: "us.amazon.nova-pro-v1:0")
        
    Returns:
        The model's response as a string
    """
    try:
        bedrock = get_bedrock_client()
        messages = []
        
        # System context - only add for the first message
        if not chat_history:
            system_prompt = f"""You are a helpful AI assistant analyzing a video. Here is the complete video analysis report:

            {video_report}

            Use this report to answer user questions about the video. Be specific, reference timestamps, and provide detailed insights.
            If asked about something not in the report, say you don't have that information."""
                
            messages.extend([
                {"role": "user", "content": [{"text": system_prompt}]},
                {"role": "assistant", "content": [{"text": "I've analyzed the video report. I'm ready to answer your questions!"}]}
            ])
        
        # Add chat history
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({
                    "role": msg["role"],
                    "content": [{"text": str(msg["content"])}]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })
        
        # Get response from Bedrock
        response = bedrock.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2048,
                "temperature": 0.7,
                "topP": 0.9,
                "stopSequences": ["\n\nHuman:"]
            }
        )
        
        # Extract and return the response text
        return response["output"]["message"]["content"][0]["text"]
        
    except Exception as e:
        error_msg = f"Error in chat_with_video: {str(e)}"
        print(error_msg)  # Log the error for debugging
        return "I'm sorry, I encountered an error processing your request. Please try again."

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "video_report" not in st.session_state:
    st.session_state.video_report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0
if "transcript_count" not in st.session_state:
    st.session_state.transcript_count = 0


# ============================================================================
# SIDEBAR - VIDEO UPLOAD & ANALYSIS
# ============================================================================

with st.sidebar:
    st.title("🎬 Video Analysis")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file to analyze"
    )
    
    if uploaded_file:
        st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    
    num_frames = st.slider(
        "Number of Frames",
        min_value=10,
        max_value=50,
        value=30,
        help="More frames = better quality but slower"
    )
    
    sample_rate = st.slider(
        "Sample Rate",
        min_value=5,
        max_value=30,
        value=20,
        help="Higher = faster processing"
    )
    
    use_lite = st.checkbox("Use Lite Model (Faster)", value=True)
    
    st.markdown("---")
    
    # Analyze button
    if uploaded_file:
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
            st.session_state.is_analyzed = False
            st.session_state.chat_history = []
            
            with st.spinner("🔄 Analyzing video... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    start_time = time.time()
                    
                    # Step 1: Extract
                    status_text.text("📹 Extracting frames and audio...")
                    progress_bar.progress(20)
                    
                    visual_analysis, transcripts, audio_clip, video_clip = analyze_video_parallel(
                        temp_path, num_frames, sample_rate
                    )
                    
                    # Step 2: Generate report
                    status_text.text("📝 Creating unified report...")
                    progress_bar.progress(70)
                    
                    model_id = "us.amazon.nova-lite-v1:0" if use_lite else "us.amazon.nova-pro-v1:0"
                    report = create_unified_report(
                        visual_analysis, transcripts, video_clip, audio_clip, model_id
                    )
                    
                    # Save to session state
                    st.session_state.video_report = report
                    st.session_state.transcript_count = len(transcripts)
                    st.session_state.processing_time = time.time() - start_time
                    st.session_state.is_analyzed = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.success(f"✅ Analysis complete in {st.session_state.processing_time:.1f}s!")
                    st.balloons()
                    
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.info("👆 Upload a video to get started")
    
    # Stats
    if st.session_state.is_analyzed:
        st.markdown("---")
        st.subheader("📊 Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Time", f"{st.session_state.processing_time:.1f}s")
        with col2:
            st.metric("Transcripts", st.session_state.transcript_count)
        
        if st.button("📥 Download Report", use_container_width=True):
            st.download_button(
                "💾 Save Markdown",
                st.session_state.video_report,
                file_name=f"video_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )


# ============================================================================
# MAIN AREA - CHAT INTERFACE
# ============================================================================

st.title("💬 Chat with Your Video")

# Check if video is analyzed
if not st.session_state.is_analyzed:
    st.markdown("""
    <div class="info-box" style="text-align: center; padding: 60px;">
        <h2>👋 Welcome!</h2>
        <p style="font-size: 18px; color: #666;">
            Upload a video in the sidebar and click "Analyze Video" to start chatting about its content.
        </p>
        <br>
        <p style="color: #999;">
            ✨ Ask about timestamps, events, highlights, and more!
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Chat interface
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 30px; color: #666;">
                <h3>👋 Ready to chat!</h3>
                <p>Ask me anything about the video:</p>
                <ul style="list-style: none; padding: 0;">
                    <li>🕐 "What happened at 2:35?"</li>
                    <li>⚽ "Summarize the key moments"</li>
                    <li>🎯 "Tell me about the goal"</li>
                    <li>📊 "What was the overall atmosphere?"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="message-sender">You</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="message-sender">AI Assistant</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about the video...",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Handle message sending
    if (send_button or user_input) and user_input.strip():
        # Get AI response FIRST (don't add user message to history yet)
        with st.spinner("🤔 AI is thinking..."):
            ai_response = chat_with_video(
                user_input,
                st.session_state.video_report,
                st.session_state.chat_history  # Pass current history WITHOUT the new message
            )
        
        # NOW add both user message and AI response to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })
        
        # Rerun to update chat
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 20px;">
    Made with ❤️ using Streamlit | 🎬 Powered by AWS Bedrock & Transcribe
</div>
""", unsafe_allow_html=True)

