import torch
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from pyvirtualdisplay import Display
from gymnasium import spaces

from source.utils import *
from source.eval import *

class MultiAgentProcgenEnv(gym.Env):
    def __init__(self, num_agents: int = 2, use_vlm: bool = False, vlm_model=None, processor=None, vlm_hidden_size=None):
        super(MultiAgentProcgenEnv, self).__init__()
        self.num_agents = num_agents
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.processor = processor
        self.vlm_hidden_size = vlm_hidden_size
        
        # Define observation space for each agent
        if use_vlm:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                'embedding': spaces.Box(low=-np.inf, high=np.inf, shape=(vlm_hidden_size,), dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            
        # Action space for each agent (movement + interaction)
        self.action_space = spaces.Discrete(15)  # Expanded action space for multi-agent interactions
        
        # Initialize agent positions and states
        self.agent_positions = []
        self.agent_scores = np.zeros(num_agents)
        self.coins = []
        self.obstacles = []
        
    def reset(self) -> Dict:
        # Reset environment state
        self.agent_positions = self._initialize_agent_positions()
        self.coins = self._initialize_coins()
        self.agent_scores = np.zeros(self.num_agents)
        
        # Generate initial observations for all agents
        observations = {}
        for agent_id in range(self.num_agents):
            if self.use_vlm:
                frame = self._render_agent_view(agent_id)
                embedding = self._get_language_embedding(frame)
                observations[f"agent_{agent_id}"] = {
                    "image": frame,
                    "embedding": embedding
                }
            else:
                observations[f"agent_{agent_id}"] = self._render_agent_view(agent_id)
                
        return observations
    
    def step(self, actions: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
        # Process actions for all agents simultaneously
        next_positions = self._compute_next_positions(actions)
        rewards = self._compute_rewards(next_positions)
        
        # Update agent positions if valid
        for agent_id, next_pos in next_positions.items():
            if self._is_valid_position(next_pos):
                self.agent_positions[agent_id] = next_pos
                
        # Check for coin collection
        self._process_coin_collection()
        
        # Generate observations for all agents
        observations = {}
        for agent_id in range(self.num_agents):
            if self.use_vlm:
                frame = self._render_agent_view(agent_id)
                embedding = self._get_language_embedding(frame)
                observations[f"agent_{agent_id}"] = {
                    "image": frame,
                    "embedding": embedding
                }
            else:
                observations[f"agent_{agent_id}"] = self._render_agent_view(agent_id)
                
        # Check if episode is done
        dones = self._check_episode_termination()
        
        # Additional info
        infos = {f"agent_{i}": {"score": self.agent_scores[i]} for i in range(self.num_agents)}
        
        return observations, rewards, dones, infos
    
    def _initialize_agent_positions(self) -> List[Tuple[int, int]]:
        # Initialize agents in different corners or random positions
        positions = []
        grid_size = 64
        corner_positions = [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]
        
        for i in range(self.num_agents):
            if i < len(corner_positions):
                positions.append(corner_positions[i])
            else:
                # Random position for additional agents
                while True:
                    pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
                    if pos not in positions:
                        positions.append(pos)
                        break
        return positions
    
    def _initialize_coins(self) -> List[Tuple[int, int]]:
        # Randomly place coins in the environment
        coins = []
        num_coins = self.num_agents * 3  # Scale coins with number of agents
        grid_size = 64
        
        for _ in range(num_coins):
            while True:
                pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
                if pos not in coins and pos not in self.agent_positions:
                    coins.append(pos)
                    break
        return coins
    
    def _compute_next_positions(self, actions: Dict) -> Dict:
        next_positions = {}
        for agent_id, action in actions.items():
            current_pos = self.agent_positions[int(agent_id.split('_')[1])]
            # Convert action to movement
            dx, dy = self._action_to_movement(action)
            next_positions[agent_id] = (current_pos[0] + dx, current_pos[1] + dy)
        return next_positions
    
    def _action_to_movement(self, action: int) -> Tuple[int, int]:
        # Convert action to movement direction
        movements = {
            0: (0, 1),   # Up
            1: (0, -1),  # Down
            2: (1, 0),   # Right
            3: (-1, 0),  # Left
            4: (1, 1),   # Up-Right
            5: (-1, 1),  # Up-Left
            6: (1, -1),  # Down-Right
            7: (-1, -1)  # Down-Left
        }
        return movements.get(action, (0, 0))
    
    def _compute_rewards(self, next_positions: Dict) -> Dict:
        rewards = {}
        for agent_id in next_positions:
            reward = 0
            next_pos = next_positions[agent_id]
            
            # Reward for collecting coins
            if next_pos in self.coins:
                reward += self.difficulty_params['coin_reward']
                
            # Penalty for colliding with other agents
            for other_id, other_pos in next_positions.items():
                if other_id != agent_id and other_pos == next_pos:
                    reward += self.difficulty_params['collision_penalty']
                    
            # Penalty for hitting obstacles
            if next_pos in self.obstacles:
                reward += self.difficulty_params['collision_penalty']
                
            rewards[agent_id] = reward
        return rewards
    
    def _process_coin_collection(self):
        # Remove collected coins and update scores
        for agent_id, pos in enumerate(self.agent_positions):
            if pos in self.coins:
                self.coins.remove(pos)
                self.agent_scores[agent_id] += 1
    
    def _check_episode_termination(self) -> Dict:
        # Episode ends when all coins are collected or max steps reached
        base_done = len(self.coins) == 0
        return {f"agent_{i}": base_done for i in range(self.num_agents)}
    
    def _render_agent_view(self, agent_id: int) -> np.ndarray:
        # Create a visual representation of the environment from agent's perspective
        grid = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Draw coins
        for coin in self.coins:
            grid[coin[0], coin[1]] = [255, 215, 0]  # Gold color
            
        # Draw agents
        for i, pos in enumerate(self.agent_positions):
            color = [0, 255, 0] if i == agent_id else [255, 0, 0]  # Green for self, red for others
            grid[pos[0], pos[1]] = color
            
        return grid
    
    def _get_language_embedding(self, frame: np.ndarray) -> np.ndarray:
        if self.use_vlm:
            inputs = self.processor(images=frame, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.vlm_model.get_image_features(**inputs)
            return outputs.cpu().numpy()
        return np.array([])

# Training setup
def create_multi_agent_environment(num_agents: int, use_vlm: bool = False):
    if use_vlm:
        model_id = "google/paligemma-3b-pt-224"
        vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        vlm_hidden_size = vlm_model.config.hidden_size
        return MultiAgentProcgenEnv(num_agents, use_vlm, vlm_model, processor, vlm_hidden_size)
    return MultiAgentProcgenEnv(num_agents, use_vlm)

# Training loop
def train_multi_agent(num_agents: int = 2, num_episodes: int = 20, use_vlm: bool = False):
    env = create_multi_agent_environment(num_agents, use_vlm)
    if use_vlm:
        # Create a vectorized environment
        env = DummyVecEnv([lambda: CustomProcgenEnv(use_vlm, vlm_hidden_size=vlm_hidden_size, vlm_model=vlm_model, processor=processor)])
    else:
        env = DummyVecEnv([lambda: CustomProcgenEnv(use_vlm)])
        
    # Create separate PPO models for each agent
    agents = {}
    for i in range(num_agents):
        if use_vlm:
            agents[f"agent_{i}"] = PPO(MultiInputActorCriticPolicy, DummyVecEnv([lambda: env]), device="cuda")
        else:
            agents[f"agent_{i}"] = PPO("CnnPolicy", DummyVecEnv([lambda: env]), device="cuda")
    
    for episode in tqdm(range(num_episodes)):
        observations = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        frames = []
        
        for step in range(100):  # Max steps per episode
            actions = {}
            for agent_id, agent in agents.items():
                action, _ = agent.predict(observations[agent_id], deterministic=True)
                actions[agent_id] = action
                
            next_observations, rewards, dones, infos = env.step(actions)
            
            # Update rewards
            for agent_id in agents:
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Store frame for visualization
            frame = env._render_agent_view(0)  # Render from first agent's perspective
            frames.append(frame)
            
            # Check if episode is done
            if any(dones.values()):
                break
                
            observations = next_observations
            
        # Save visualization every few episodes
        if episode % 5 == 0:
            save_video(frames, f'examples/multi_agent_episode_{episode}.mp4', fps=30)
            
        print(f"Episode {episode} rewards:", episode_rewards)
        
    return agents

if __name__ == "__main__":
    # Set up environment variables and display
    os.environ['XDG_RUNTIME_DIR'] = '/tmp'
    display = Display(visible=0, size=(800, 600))
    display.start()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    
    # Train multi-agent system
    trained_agents = train_multi_agent(num_agents=3, num_episodes=20, use_vlm=False)