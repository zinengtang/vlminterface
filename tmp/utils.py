import torch
import cv2
import numpy as np
import gym
from PIL import Image as PILImage
from gym import spaces


def save_video(frames, filename, fps=30):
    """
    Save a list of frames as an MP4 video, excluding any frames that are None.
    
    Args:
        frames (list): List of frames (numpy arrays) to save as a video.
        filename (str): Output filename for the video, e.g., 'output.mp4'.
        fps (int): Frames per second for the video.
    """
    # Filter out None frames
    valid_frames = [frame for frame in frames if frame is not None]

    if valid_frames:
        height, width, _ = valid_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in valid_frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"Video saved as {filename}")
    else:
        print("No valid frames to save.")


def get_language_embedding(observation_frame, vlm_model, processor):
    """Generate a language embedding based on the current observation frame using VLM."""
    # Convert the observation to a PIL Image
    image = PILImage.fromarray(observation_frame)

    # Prepare a prompt for the VLM model
    prompt = "<image> <bos> Describe this environment in details."
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(vlm_model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    # Generate output with the VLM model
    with torch.no_grad():
        generation = vlm_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generated_text = processor.decode(generation[0][input_len:], skip_special_tokens=True)

        # Extract the embedding from generated tokens
        generation = generation[0][input_len:]
        embedding = vlm_model.get_input_embeddings()(generation)
        language_embedding = embedding.mean(dim=0).cpu().numpy()  # Pooling and convert to numpy

    return language_embedding, generated_text

# Custom environment wrapper to output Dict observations when using VLM
class CustomProcgenEnv(gym.Env):
    def __init__(self, use_vlm, vlm_hidden_size=None, vlm_model=None, processor=None):
        super(CustomProcgenEnv, self).__init__()
        self.env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1, render_mode='rgb_array')
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.processor = processor
        
        if use_vlm:
            # Define observation space for image and embedding separately
            self.observation_space = spaces.Dict({
                "image": self.env.observation_space,  # Original observation space
                "embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(vlm_hidden_size,), dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Dict({
                "image": self.env.observation_space,  # Original observation space
                # "embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(vlm_hidden_size,), dtype=np.float32)
            })
            # self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space

    def reset(self):
        obs = self.env.reset()
        if self.use_vlm:
            current_frame = obs.astype(np.uint8)
            language_embedding, generated_text = get_language_embedding(current_frame, self.vlm_model, self.processor)
            return {
                "image": obs,
                "embedding": language_embedding,
                "generated_text": generated_text
            }
        else:
            return {
                "image": obs,
            }

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.use_vlm:
            current_frame = obs.astype(np.uint8)
            language_embedding, generated_text = get_language_embedding(current_frame, self.vlm_model, self.processor)
            return {
                "image": obs,
                "embedding": language_embedding,
                "generated_text": generated_text
            }, reward, done, info
        else:
            return {
                "image": obs,
            }, reward, done, info

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
