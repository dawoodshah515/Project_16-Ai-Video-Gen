import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
import tempfile
import base64
from typing import List, Dict, Any
import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import google.generativeai as genai
import re
from groq import Groq

# Load environment variables
load_dotenv()

# Constants
VIDEOS_FOLDER = "Videos"
DEFAULT_TARGET_DURATION = 30  # seconds
VIDEO_WIDTH = 640  # Reduced width for better display
VIDEO_HEIGHT = 360  # Reduced height for better display (16:9 aspect ratio)

# Configure page
st.set_page_config(
    page_title="VidGen AI - Professional Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black and blue theme
def set_custom_style():
    st.markdown("""
    <style>
        :root {
            --primary-color: #0066cc;
            --secondary-color: #003d7a;
            --accent-color: #0099ff;
            --background-color: #0a0a0a;
            --card-background: #1a1a1a;
            --text-color: #ffffff;
            --text-secondary: #b0b0b0;
        }
        
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 800px;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color);
        }
        
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            width: 100%;
            margin: 0.5rem 0;
        }
        
        .stButton>button:hover {
            background-color: var(--accent-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 153, 255, 0.3);
        }
        
        .stTextArea>div>div>textarea {
            background-color: var(--card-background);
            color: var(--text-color);
            border: 2px solid var(--primary-color);
            border-radius: 8px;
            font-size: 1rem;
            padding: 1rem;
        }
        
        .stTextArea>div>div>textarea::placeholder {
            color: var(--text-secondary);
        }
        
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
        }
        
        .loading-text {
            color: var(--text-color);
            margin-top: 1.5rem;
            font-size: 1.3rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: var(--card-background);
            border-radius: 4px;
            margin-top: 1.5rem;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .video-container {
            background-color: var(--card-background);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 2px solid var(--primary-color);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            max-width: 100%;
            overflow: hidden;
        }
        
        .video-container video {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        
        .header-container {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .api-status {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 1rem;
        }
        
        .api-active {
            background-color: rgba(0, 255, 0, 0.2);
            color: #00ff00;
            border: 1px solid #00ff00;
        }
        
        .api-inactive {
            background-color: rgba(255, 0, 0, 0.2);
            color: #ff3333;
            border: 1px solid #ff3333;
        }
        
        .success-message {
            background-color: rgba(0, 255, 0, 0.1);
            border-left: 4px solid #33ff33;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: var(--text-color);
        }
        
        .error-message {
            background-color: rgba(255, 0, 0, 0.1);
            border-left: 4px solid #ff3333;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: var(--text-color);
        }
    </style>
    """, unsafe_allow_html=True)

def create_loading_animation():
    """Create a circular blue dots loading animation"""
    frames = []
    num_dots = 8
    dot_radius = 12
    circle_radius = 60
    
    for i in range(30):
        img = Image.new('RGBA', (250, 250), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        for j in range(num_dots):
            angle = (j / num_dots) * 2 * np.pi + (i * 0.1)
            x = 125 + circle_radius * np.cos(angle)
            y = 125 + circle_radius * np.sin(angle)
            
            opacity = int(255 * (0.5 + 0.5 * np.sin(angle + i * 0.2)))
            color = (0, 102, 204, opacity)
            
            draw.ellipse(
                [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
                fill=color
            )
        
        frames.append(img)
    
    with io.BytesIO() as output:
        frames[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )
        return base64.b64encode(output.getvalue()).decode()

def get_video_files() -> List[str]:
    """Get list of available video files from the Videos folder"""
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
        return []
    
    video_files = []
    for filename in os.listdir(VIDEOS_FOLDER):
        if filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            clause_name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ')
            video_files.append((filename, clause_name))
    
    return video_files

def find_matching_videos(prompt: str, video_files: List[tuple]) -> List[str]:
    """Find video files that match the prompt"""
    prompt = prompt.lower().strip()
    matching_videos = []
    
    for filename, clause_name in video_files:
        clause_name = clause_name.lower()
        
        # Check for exact match
        if prompt == clause_name:
            matching_videos.append(filename)
            continue
        
        # Check for partial match
        words_in_prompt = set(prompt.split())
        words_in_name = set(clause_name.split())
        
        # Calculate similarity score
        intersection = words_in_prompt.intersection(words_in_name)
        if len(intersection) > 0:
            score = len(intersection) / max(len(words_in_prompt), len(words_in_name))
            if score > 0.3:  # Threshold for matching
                matching_videos.append(filename)
    
    return matching_videos

def generate_video_plan_with_gemini(prompt: str, target_duration: float) -> Dict[str, Any]:
    """Generate a video plan using Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt_text = f"""
        Create a video plan for the prompt: "{prompt}"
        
        Generate JSON with this structure:
        {{
            "title": "Short title",
            "clauses": [
                {{
                    "clause_name": "matching phrase",
                    "duration_sec": 5.0,
                    "transition_to_next": "crossfade"
                }}
            ]
        }}
        
        Total duration: {target_duration} seconds
        Each clause: 1.5-8.0 seconds
        Respond with valid JSON only.
        """
        
        response = model.generate_content(prompt_text)
        plan_text = response.text
        
        # Extract JSON
        json_match = re.search(r'```json\n(.*?)\n```', plan_text, re.DOTALL)
        if json_match:
            plan_json = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan_json = json_match.group(0)
            else:
                plan_json = plan_text
        
        plan = json.loads(plan_json)
        return plan
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return None

def generate_video_plan_with_groq(prompt: str, target_duration: float) -> Dict[str, Any]:
    """Generate a video plan using Groq API"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    
    try:
        client = Groq(api_key=api_key)
        
        prompt_text = f"""
        Create a video plan for the prompt: "{prompt}"
        
        Generate JSON with this structure:
        {{
            "title": "Short title",
            "clauses": [
                {{
                    "clause_name": "matching phrase",
                    "duration_sec": 5.0,
                    "transition_to_next": "crossfade"
                }}
            ]
        }}
        
        Total duration: {target_duration} seconds
        Each clause: 1.5-8.0 seconds
        Respond with valid JSON only.
        """
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.5,
            max_tokens=1024
        )
        
        plan_text = response.choices[0].message.content
        
        # Extract JSON
        json_match = re.search(r'```json\n(.*?)\n```', plan_text, re.DOTALL)
        if json_match:
            plan_json = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan_json = json_match.group(0)
            else:
                plan_json = plan_text
        
        plan = json.loads(plan_json)
        return plan
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        return None

def generate_video_plan_with_openai(prompt: str, target_duration: float) -> Dict[str, Any]:
    """Generate a video plan using OpenAI API as another fallback"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        prompt_text = f"""
        Create a video plan for the prompt: "{prompt}"
        
        Generate JSON with this structure:
        {{
            "title": "Short title",
            "clauses": [
                {{
                    "clause_name": "matching phrase",
                    "duration_sec": 5.0,
                    "transition_to_next": "crossfade"
                }}
            ]
        }}
        
        Total duration: {target_duration} seconds
        Each clause: 1.5-8.0 seconds
        Respond with valid JSON only.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.5,
            max_tokens=1024
        )
        
        plan_text = response.choices[0].message.content
        
        # Extract JSON
        json_match = re.search(r'```json\n(.*?)\n```', plan_text, re.DOTALL)
        if json_match:
            plan_json = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan_json = json_match.group(0)
            else:
                plan_json = plan_text
        
        plan = json.loads(plan_json)
        return plan
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return None

def generate_video_plan_smart(prompt: str, target_duration: float) -> tuple:
    """Try Gemini first, then Groq, then OpenAI if previous ones fail"""
    # Try Gemini first
    plan = generate_video_plan_with_gemini(prompt, target_duration)
    if plan:
        return plan, "Gemini"
    
    # Try Groq as fallback
    plan = generate_video_plan_with_groq(prompt, target_duration)
    if plan:
        return plan, "Groq"
    
    # Try OpenAI as another fallback
    plan = generate_video_plan_with_openai(prompt, target_duration)
    if plan:
        return plan, "OpenAI"
    
    # If all fail, return None
    return None, "None"

def create_video_from_prompt(prompt: str, output_path: str, target_duration: float, progress_callback=None) -> str:
    """Create a video based on the prompt"""
    # Get available video files
    video_files = get_video_files()
    
    if not video_files:
        return create_placeholder_video(output_path, target_duration, "No video clips available\nPlease add clips to the Videos folder")
    
    # Generate plan using smart API selection
    plan, api_used = generate_video_plan_smart(prompt, target_duration)
    
    if not plan:
        # Create a simple plan if all APIs fail
        matching_videos = find_matching_videos(prompt, video_files)
        if not matching_videos:
            return create_placeholder_video(output_path, target_duration, "No matching video clips found")
        
        plan = {
            "title": "Generated Video",
            "clauses": []
        }
        
        duration_per_clip = target_duration / len(matching_videos)
        for i, filename in enumerate(matching_videos):
            clause_name = next(name for f, name in video_files if f == filename)
            plan["clauses"].append({
                "clause_name": clause_name,
                "duration_sec": max(1.5, min(8.0, duration_per_clip)),
                "transition_to_next": "crossfade"
            })
    
    # Load video clips
    clips = []
    total_clauses = len(plan["clauses"])
    
    for i, clause_info in enumerate(plan["clauses"]):
        clause_name = clause_info["clause_name"]
        
        # Find the matching video file
        matching_file = None
        for filename, name in video_files:
            if name.lower() == clause_name.lower():
                matching_file = filename
                break
        
        if matching_file:
            filepath = os.path.join(VIDEOS_FOLDER, matching_file)
            
            if os.path.exists(filepath):
                clip = mp.VideoFileClip(filepath)
                # Resize clip to our target dimensions
                clip = clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
                
                if clip.duration > clause_info["duration_sec"]:
                    clip = clip.subclip(0, clause_info["duration_sec"])
                clips.append(clip)
        
        if progress_callback:
            progress = (i + 1) / (total_clauses + 1)
            progress_callback(progress)
    
    if not clips:
        return create_placeholder_video(output_path, target_duration, "Error loading video clips")
    
    # Concatenate clips
    try:
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            bitrate="1000k",  # Lower bitrate for smaller file size
            fps=24  # Standard frame rate
        )
        
        for clip in clips:
            clip.close()
        final_clip.close()
    except Exception as e:
        return create_placeholder_video(output_path, target_duration, f"Error creating video: {str(e)}")
    
    if progress_callback:
        progress_callback(1.0)
    
    return output_path

def create_placeholder_video(output_path: str, duration: float, message: str) -> str:
    """Create a placeholder video with text"""
    try:
        txt_clip = mp.TextClip(message, fontsize=30, color='white', bg_color='black')
        txt_clip = txt_clip.set_duration(duration).set_position('center').set_fps(24)
        txt_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            bitrate="1000k",
            fps=24
        )
        txt_clip.close()
        return output_path
    except:
        # Fallback to simple placeholder
        width, height = VIDEO_WIDTH, VIDEO_HEIGHT  # Use our defined dimensions
        fps = 24
        frames = int(duration * fps)
        
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(black_frame)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        lines = message.split('\n')
        y_offset = (height - len(lines) * 40) // 2
        
        for line in lines:
            if font:
                # Use textbbox instead of textsize (compatible with newer Pillow versions)
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (width - text_width) // 2
                draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += text_height + 10
        
        frame = np.array(img)
        frame_list = [frame for _ in range(frames)]
        
        clip = mp.ImageSequenceClip(frame_list, fps=fps)
        clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            bitrate="1000k",
            fps=24
        )
        clip.close()
        
        return output_path

def show_progress_bar(progress):
    """Display a custom progress bar"""
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress * 100}%"></div>
    </div>
    """, unsafe_allow_html=True)

def check_api_status():
    """Check which APIs are available"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    gemini_status = "Active" if gemini_key else "Inactive"
    groq_status = "Active" if groq_key else "Inactive"
    openai_status = "Active" if openai_key else "Inactive"
    
    return gemini_status, groq_status, openai_status

def main():
    set_custom_style()
    loading_animation = create_loading_animation()
    
    # Header with API status
    gemini_status, groq_status, openai_status = check_api_status()
    
    st.markdown(f"""
    <div class="header-container">
        <h1 style="margin: 0; color: #0099ff; font-size: 3rem;">VidGen AI</h1>
        <p style="color: var(--text-secondary); font-size: 1.2rem; margin-top: 0.5rem;">Professional Video Generator</p>
        <div style="margin-top: 1rem;">
            <span class="api-status api-{gemini_status.lower()}">Gemini: {gemini_status}</span>
            <span class="api-status api-{groq_status.lower()}">Groq: {groq_status}</span>
            <span class="api-status api-{openai_status.lower()}">OpenAI: {openai_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Videos folder exists
    if not os.path.exists(VIDEOS_FOLDER):
        os.makedirs(VIDEOS_FOLDER)
    
    # Get available video files
    video_files = get_video_files()
    
    # Main prompt input
    prompt = st.text_area(
        "enter prompt to generate a video",
        "Hello, how are you today? I hope you're having a wonderful day!",
        height=100
    )
    
    # Generate button
    generate_button = st.button("🎬 Generate Video", type="primary")
    
    # Video generation
    if generate_button:
        if not prompt.strip():
            st.error("Please enter a prompt.")
            return
        
        if not video_files:
            st.markdown("""
            <div class="error-message">
                <strong>No video clips found!</strong><br>
                Please add video clips to the Videos folder. Name them after spoken phrases (e.g., 'hello_how_are_you.mp4')
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Show loading animation
        placeholder = st.empty()
        
        with placeholder.container():
            st.markdown(f"""
            <div class="loading-container">
                <img src="data:image/gif;base64,{loading_animation}" alt="Loading...">
                <div class="loading-text">Generating your video...</div>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.empty()
            show_progress_bar(0)
        
        # Create video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            output_path = tmp_file.name
        
        def update_progress(progress):
            progress_bar.empty()
            with progress_bar.container():
                show_progress_bar(progress)
        
        try:
            create_video_from_prompt(prompt, output_path, DEFAULT_TARGET_DURATION, update_progress)
            
            # Clear loading animation
            placeholder.empty()
            
            # Success message
            st.markdown("""
            <div class="success-message">
                <strong>✅ Video generated successfully!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Display video with custom container for better sizing
            st.markdown("""
            <div class="video-container">
                <h2 style="color: #0099ff; margin-top: 0;">Your Generated Video</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display video with width control
            st.video(output_path, format="video/mp4", start_time=0)
            
            # Download button
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name="generated_video.mp4",
                mime="video/mp4"
            )
            
        except Exception as e:
            placeholder.empty()
            st.markdown(f"""
            <div class="error-message">
                <strong>Error generating video:</strong><br>
                {str(e)}
            </div>
            """, unsafe_allow_html=True)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

if __name__ == "__main__":
    main()