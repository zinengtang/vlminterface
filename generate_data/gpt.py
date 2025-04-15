import cv2
import openai
import base64
import io
from PIL import Image

# Replace with your OpenAI API key.
openai.api_key = 'YOUR_API_KEY'

def image_to_base64(image):
    """
    Convert a single video frame (as a NumPy array) into a base64-encoded JPEG.
    This string could be used to represent the image in a text prompt.
    """
    # Convert the BGR image (from OpenCV) to RGB for Pillow.
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    # Encode the binary image data to a base64 string.
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def annotate_frame(frame):
    """
    Annotate a single video frame by sending a prompt to the OpenAI API.
    The prompt includes a (truncated) base64 string representing the image,
    along with a request for the model to identify the action in the video game frame.
    
    Note: This is a conceptual approach since OpenAI's current public API
    does not fully support direct image interpretation.
    """
    # Convert the frame to a base64 string.
    encoded_image = image_to_base64(frame)
    
    # Build a prompt. For example, we include a short snippet of the encoded image.
    # In a real-world scenario, you might want to integrate a dedicated image captioning
    # system or wait for vision-enabled API endpoints.
    prompt = (
        "You are a video game analyst. Look at the provided image representation "
        "of a video game frame (given as a base64 string snippet) and describe the action happening in medium level. "
        "For example, the character might be 'moving to a platform', 'retrieving the key'. "
        "Image snippet: " + encoded_image[:100] + "..."
    )
    
    # Call the OpenAI ChatCompletion API with the prompt.
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use "gpt-3.5-turbo" if preferred
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    
    # Extract the reply from the API response.
    annotation = response.choices[0].message['content']
    return annotation

def annotate_video(video_path, frame_sample_rate=30):
    """
    Process a video file, sample frames at the specified rate (default every 30 frames),
    and obtain an annotation (action description) for each sampled frame using the annotate_frame function.
    
    Returns:
        A dictionary mapping the frame number to its annotation.
    """
    # Open the video file using OpenCV.
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    annotations = {}
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return annotations
    
    # Process frames from the video.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 'frame_sample_rate'-th frame.
        if frame_count % frame_sample_rate == 0:
            print(f"Annotating frame {frame_count}...")
            annotation = annotate_frame(frame)
            annotations[frame_count] = annotation
            print(f"Frame {frame_count} annotation: {annotation}\n")
        
        frame_count += 1

    cap.release()
    return annotations

def generate_high_level_actions(low_level_actions):
    """
    Generate a higher-level description or coherent sequence of actions from a list of low-level actions.
    
    The function accepts a list of strings where each string is a low-level action (e.g., 'moved left',
    'fired weapon', 'jumped') and constructs a prompt to generate a higher-level summary using the OpenAI API.
    
    Returns:
        A string with the generated high-level action description.
    """
    # Create a descriptive prompt that incorporates the low-level actions.
    prompt = (
        "You are a video game analyst. Given the following list of low-level actions performed in a video game, "
        "generate a medium level actions that describes the overall gameplay sequence every 0 to 5 seconds. "
        "Low-level actions: " + ", ".join(low_level_actions)
    )
    
    # Call the OpenAI ChatCompletion API with the prompt.
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use "gpt-3.5-turbo" if desired
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    
    # Extract and return the generated action summary.
    action_summary = response.choices[0].message['content']
    return action_summary
