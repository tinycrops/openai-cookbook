import os
import io
import base64
import asyncio
import datetime
from datetime import UTC  # Use timezone-aware objects
import sqlite3
import json
from typing import Optional, Dict, Any, List, Tuple
import random
import time
import copy
import collections
import math
import warnings

import cv2
import PIL.Image
from google import genai
from google.genai import types
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural network world model will be disabled.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Performance visualization will be disabled.")

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    warnings.warn("Spotipy not available. Spotify integration will be disabled.")

from agentic_info_systems.signal_recorder import SignalRecorder

# --- CONFIGURATION ---

DB_PATH = "agentic_insights.db"
MODEL = "models/gemini-2.0-flash"  # Fixed model name for live API
EMBED_MODEL = "text-embedding-004"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CYCLE_WAIT_TIME = 15  # Seconds to wait between interaction cycles

# Spotify configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8080")

# RL configuration
EXPERIENCE_BUFFER_SIZE = 10000
EMBEDDING_DIM = 768  # Dimension for state/action embeddings
GAMMA = 0.95  # Discount factor for future rewards
EPSILON_START = 0.3  # Start exploration rate
EPSILON_MIN = 0.05  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration
LEARNING_RATE = 1e-4  # Learning rate for neural networks
BATCH_SIZE = 16  # Batch size for training
TARGET_UPDATE_FREQ = 5  # How often to update target network

# Initialize signal recorder for monitoring metrics
signal_recorder = SignalRecorder(db_path='agentic_info_systems/signals.db')

client = genai.Client(api_key=GEMINI_API_KEY)

# --- NEURAL NETWORK DEFINITIONS ---

class ExperienceReplay:
    """Experience replay buffer for RL algorithms"""
    
    def __init__(self, capacity=EXPERIENCE_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

if TORCH_AVAILABLE:
    class StateActionEncoder(nn.Module):
        """Neural encoder for state and action embeddings"""
        
        def __init__(self, input_dim, hidden_dim=128, output_dim=EMBEDDING_DIM):
            super(StateActionEncoder, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    class QNetwork(nn.Module):
        """Q-Network for estimating state-action values"""
        
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            
        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    class NeuralWorldModel(nn.Module):
        """Neural network world model for predicting next state and reward"""
        
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(NeuralWorldModel, self).__init__()
            # Shared layers
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            
            # State prediction branch
            self.state_fc = nn.Linear(hidden_dim, hidden_dim)
            self.state_out = nn.Linear(hidden_dim, state_dim)
            
            # Reward prediction branch
            self.reward_fc = nn.Linear(hidden_dim, hidden_dim)
            self.reward_out = nn.Linear(hidden_dim, 1)
            
        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            
            # State prediction
            s = F.relu(self.state_fc(x))
            next_state = self.state_out(s)
            
            # Reward prediction
            r = F.relu(self.reward_fc(x))
            reward = self.reward_out(r)
            
            return next_state, reward

# --- DATABASE SETUP ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Main observations table with face embeddings
    c.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            image BLOB,
            insights TEXT,
            agent_state TEXT,
            face_description TEXT,
            face_embedding BLOB
        )
    """)
    
    # User embeddings for identification
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_embeddings (
            user_id TEXT PRIMARY KEY,
            avg_embedding BLOB,
            count INTEGER
        )
    """)
    
    # New table for actions taken by the agent
    c.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_id INTEGER,
            timestamp TEXT,
            action_type TEXT,
            action_content TEXT,
            FOREIGN KEY (observation_id) REFERENCES observations(id)
        )
    """)
    
    # New table for rewards and feedback
    c.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id INTEGER,
            timestamp TEXT,
            reward_value REAL,
            reward_source TEXT,
            context TEXT,
            FOREIGN KEY (action_id) REFERENCES actions(id)
        )
    """)
    
    # New table for world model predictions
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            state_before TEXT,
            predicted_action TEXT,
            predicted_next_state TEXT,
            predicted_reward REAL,
            actual_next_state TEXT,
            actual_reward REAL
        )
    """)
    
    # New table for reward preferences
    c.execute("""
        CREATE TABLE IF NOT EXISTS reward_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            reward_type TEXT,
            weight REAL,
            UNIQUE(user_id, reward_type)
        )
    """)
    
    # New table for state and action embeddings
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            type TEXT, /* 'state' or 'action' */
            raw_data TEXT,
            embedding BLOB,
            timestamp TEXT
        )
    """)
    
    # New table for model weights
    c.execute("""
        CREATE TABLE IF NOT EXISTS model_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            model_name TEXT, /* 'q_network', 'target_q_network', 'world_model' */
            weights BLOB,
            timestamp TEXT,
            performance_metrics TEXT
        )
    """)
    
    # New table for Spotify playlist history
    c.execute("""
        CREATE TABLE IF NOT EXISTS spotify_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action_id INTEGER,
            playlist_id TEXT,
            playlist_name TEXT,
            timestamp TEXT,
            user_skipped BOOLEAN,
            FOREIGN KEY (action_id) REFERENCES actions(id)
        )
    """)
    
    # New table for experimentation results
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            experiment_name TEXT,
            start_time TEXT,
            end_time TEXT,
            config TEXT,
            results TEXT,
            notes TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_observation(user_id: str, image_bytes: bytes, insights: str, agent_state: str, face_description: str, face_embedding: np.ndarray) -> int:
    """Save observation and return the observation ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Convert face_embedding to BLOB
    embedding_blob = face_embedding.tobytes()
    
    timestamp = datetime.datetime.now(UTC).isoformat()
    c.execute(
        "INSERT INTO observations (user_id, timestamp, image, insights, agent_state, face_description, face_embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, timestamp, image_bytes, insights, agent_state, face_description, embedding_blob)
    )
    observation_id = c.lastrowid
    conn.commit()
    conn.close()
    return observation_id

def save_action(observation_id: int, action_type: str, action_content: str) -> int:
    """Save an action taken by the agent and return the action ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.datetime.now(UTC).isoformat()
    c.execute(
        "INSERT INTO actions (observation_id, timestamp, action_type, action_content) VALUES (?, ?, ?, ?)",
        (observation_id, timestamp, action_type, action_content)
    )
    action_id = c.lastrowid
    conn.commit()
    conn.close()
    return action_id

def save_reward(action_id: int, reward_value: float, reward_source: str, context: str = "") -> int:
    """Save a reward received for an action"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.datetime.now(UTC).isoformat()
    c.execute(
        "INSERT INTO rewards (action_id, timestamp, reward_value, reward_source, context) VALUES (?, ?, ?, ?, ?)",
        (action_id, timestamp, reward_value, reward_source, context)
    )
    reward_id = c.lastrowid
    conn.commit()
    conn.close()
    return reward_id

def save_prediction(state_before: str, predicted_action: str, predicted_next_state: str, 
                   predicted_reward: float, actual_next_state: str = None, actual_reward: float = None):
    """Save a world model prediction and optionally its actual outcomes"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.datetime.now(UTC).isoformat()
    c.execute(
        "INSERT INTO predictions (timestamp, state_before, predicted_action, predicted_next_state, predicted_reward, actual_next_state, actual_reward) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timestamp, state_before, predicted_action, predicted_next_state, predicted_reward, actual_next_state, actual_reward)
    )
    conn.commit()
    conn.close()

def get_user_reward_preferences(user_id: str) -> Dict[str, float]:
    """Get user's reward preferences as a dictionary of reward_type -> weight"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT reward_type, weight FROM reward_preferences WHERE user_id = ?", (user_id,))
    preferences = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    
    # Default preferences if none are set
    if not preferences:
        preferences = {
            "mood_improvement": 1.0,
            "engagement": 0.8,
            "novelty": 0.5
        }
        # Save defaults
        set_user_reward_preferences(user_id, preferences)
    
    return preferences

def set_user_reward_preferences(user_id: str, preferences: Dict[str, float]):
    """Set or update user's reward preferences"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for reward_type, weight in preferences.items():
        c.execute(
            "INSERT INTO reward_preferences (user_id, reward_type, weight) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, reward_type) DO UPDATE SET weight = ?",
            (user_id, reward_type, weight, weight)
        )
    
    conn.commit()
    conn.close()

def get_last_insight(user_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, timestamp, insights, agent_state FROM observations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "timestamp": row[1], "insights": row[2], "agent_state": row[3]}
    return None

def get_historical_interactions(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent observations, actions, and rewards for a user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT o.id, o.timestamp, o.insights, a.action_type, a.action_content, r.reward_value, r.reward_source
        FROM observations o
        LEFT JOIN actions a ON o.id = a.observation_id
        LEFT JOIN rewards r ON a.id = r.action_id
        WHERE o.user_id = ?
        ORDER BY o.timestamp DESC
        LIMIT ?
        """,
        (user_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "observation_id": row[0],
            "timestamp": row[1],
            "insights": row[2],
            "action_type": row[3],
            "action_content": row[4],
            "reward_value": row[5],
            "reward_source": row[6]
        })
    
    return history

# --- IMAGE CAPTURE ---

def capture_image() -> bytes:
    """Capture an image from the webcam and return as JPEG bytes"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(frame_rgb)
    img.thumbnail([1024, 1024])
    image_io = io.BytesIO()
    img.save(image_io, format="jpeg")
    return image_io.getvalue()

# --- GEMINI INSIGHT EXTRACTION ---

async def extract_insights(image_bytes: bytes, last_insight: Optional[Dict[str, Any]], history: List[Dict[str, Any]]) -> str:
    """Extract insights from image with historical context"""
    system_instruction_text = (
        "You are an agentic assistant that learns from experience. "
        "Given a photo of a user, extract multi-level insights: "
        "1. Describe the user's appearance, mood, and environment objectively. "
        "2. Compare with previous observations and note any changes. "
        "3. Infer the user's current state (focused, distracted, energetic, tired, etc). "
        "4. Consider what might help the user right now based on observed patterns. "
        "Be concise, specific, and use quantifiable observations when possible."
    )

    # Format history for context
    history_context = ""
    if history:
        history_context = "\n\nRecent interaction history:\n"
        for entry in history[:3]:  # Use the most recent 3 entries
            h_timestamp = entry.get("timestamp", "Unknown time")
            h_insights = entry.get("insights", "No insights")
            h_action = entry.get("action_content", "No action")
            h_reward = entry.get("reward_value", "No reward")
            
            history_context += f"- Time: {h_timestamp}\n"
            history_context += f"  Insights: {h_insights[:100]}...\n"
            history_context += f"  Action: {h_action}\n"
            history_context += f"  Observed response: {h_reward}\n"

    # Combine instruction and prompt
    prompt_text = f"{system_instruction_text}\n\nAnalyze the user in this image.{history_context}"
    
    user_parts = [
        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)),
        types.Part(text=prompt_text)
    ]

    if last_insight:
         # Append last insight context
         user_parts.append(
              types.Part(text=
                  f"\n\nContext from previous observation ({last_insight['timestamp']}):\n{last_insight['insights']}"
              )
         )

    try:
        # Use Gemini to generate insights
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=user_parts
        )
        
        # Extract text from response
        if hasattr(response, 'text') and response.text:
             return response.text.strip()
        elif response.candidates and response.candidates[0].content.parts:
             return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
        else:
             print("Warning: No text found in response.")
             return "No insights available." 

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return f"Error generating insights: {str(e)}"

# --- AGENT STATE MANAGEMENT ---

class AgentState:
    """Structured agent state that persists across interactions"""
    
    def __init__(self, user_id: str, load_from_db: bool = True):
        self.user_id = user_id
        self.observations = []
        self.actions = []
        self.rewards = []
        self.mood_trend = []
        self.engagement_trend = []
        self.last_observation_time = None
        self.total_reward = 0.0
        self.successful_strategies = {}
        
        if load_from_db:
            self.load_from_db()
    
    def load_from_db(self):
        """Load state from database"""
        # Get recent history and extract trends
        history = get_historical_interactions(self.user_id, limit=10)
        
        if history:
            # Extract mood and engagement trends
            self.mood_trend = [h.get("mood_value", 0.0) for h in history if "mood_value" in h]
            self.engagement_trend = [h.get("reward_value", 0.0) for h in history if "reward_value" in h]
            
            # Calculate which action types tend to work best
            action_rewards = {}
            for entry in history:
                action_type = entry.get("action_type")
                reward = entry.get("reward_value")
                if action_type and reward is not None:
                    if action_type not in action_rewards:
                        action_rewards[action_type] = []
                    action_rewards[action_type].append(reward)
            
            # Find most successful strategies
            for action_type, rewards in action_rewards.items():
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    self.successful_strategies[action_type] = avg_reward
    
    def to_json(self) -> str:
        """Convert state to JSON string"""
        state_dict = {
            "user_id": self.user_id,
            "total_observations": len(self.observations),
            "total_actions": len(self.actions),
            "total_reward": self.total_reward,
            "mood_trend": self.mood_trend[-5:] if self.mood_trend else [],
            "engagement_trend": self.engagement_trend[-5:] if self.engagement_trend else [],
            "successful_strategies": self.successful_strategies,
            "last_observation_time": self.last_observation_time
        }
        return json.dumps(state_dict)
    
    def update(self, new_observation: Dict[str, Any] = None, 
              new_action: Dict[str, Any] = None, 
              new_reward: float = None,
              new_mood: float = None):
        """Update state with new information"""
        if new_observation:
            self.observations.append(new_observation)
            self.last_observation_time = datetime.datetime.now(UTC).isoformat()
        
        if new_action:
            self.actions.append(new_action)
        
        if new_reward is not None:
            self.rewards.append(new_reward)
            self.total_reward += new_reward
            
            # Update successful strategies if we have both action and reward
            if new_action and new_reward is not None:
                action_type = new_action.get("type", "unknown")
                if action_type not in self.successful_strategies:
                    self.successful_strategies[action_type] = new_reward
                else:
                    # Moving average
                    self.successful_strategies[action_type] = 0.8 * self.successful_strategies[action_type] + 0.2 * new_reward
        
        if new_mood is not None:
            self.mood_trend.append(new_mood)
            # Keep only the most recent 20 entries
            if len(self.mood_trend) > 20:
                self.mood_trend = self.mood_trend[-20:]

# --- FACE DESCRIPTION GENERATION ---

async def generate_face_description(image_bytes: bytes) -> str:
    """Generate textual description of face for embedding"""
    prompt = (
        "Describe the face in this image for the purpose of recognizing the same person in the future, "
        "even if their appearance changes (e.g., beard/no beard, glasses/no glasses). "
        "Focus on stable features like face shape, eyes, etc. Be concise but specific."
    )
    user_parts = [
        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)),
        types.Part(text=prompt)
    ]
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=user_parts
    )
    return response.text.strip() if hasattr(response, 'text') else ""

# --- FACE RECOGNITION & EMBEDDING ---

def get_face_embedding(face_description: str) -> np.ndarray:
    """Convert face description to embedding using Gemini embedding API"""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=face_description
    )
    emb_obj = response.embeddings[0]
    # Try common attributes for embedding vector
    if hasattr(emb_obj, 'values'):
        emb = emb_obj.values
    elif hasattr(emb_obj, 'embedding'):
        emb = emb_obj.embedding
    else:
        print(f"Unknown embedding object structure: {type(emb_obj)}; dir: {dir(emb_obj)}")
        raise ValueError("Cannot extract embedding vector from ContentEmbedding object.")
    return np.array(emb, dtype=np.float32)

def save_user_embedding(user_id: str, new_embedding: np.ndarray):
    """Update average embedding for a user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT avg_embedding, count FROM user_embeddings WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    
    # Ensure embedding is in correct format (binary)
    embedding_blob = new_embedding.tobytes()
    
    if row:
        avg_emb_data = row[0]
        count = row[1]
        
        # Handle both string and binary formats from database
        if isinstance(avg_emb_data, bytes):
            avg_emb = np.frombuffer(avg_emb_data, dtype=np.float32)
        else:
            # Handle legacy string format
            try:
                avg_emb = np.fromstring(avg_emb_data, sep=',', dtype=np.float32)
            except (ValueError, TypeError):
                # In case of error, just use the new embedding
                avg_emb = new_embedding
                count = 0
        
        updated_emb = (avg_emb * count + new_embedding) / (count + 1)
        c.execute("UPDATE user_embeddings SET avg_embedding = ?, count = ? WHERE user_id = ?",
                  (updated_emb.tobytes(), count + 1, user_id))
    else:
        c.execute("INSERT INTO user_embeddings (user_id, avg_embedding, count) VALUES (?, ?, ?)",
                  (user_id, embedding_blob, 1))
    
    conn.commit()
    conn.close()

def get_all_user_embeddings():
    """Get all user embeddings for identification"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, avg_embedding FROM user_embeddings")
    rows = c.fetchall()
    conn.close()
    
    user_embs = []
    for user_id, emb_blob in rows:
        if emb_blob:
            try:
                # Handle both binary blob and string format
                if isinstance(emb_blob, bytes):
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                else:
                    # Handle legacy string format
                    emb = np.fromstring(emb_blob, sep=',', dtype=np.float32)
                user_embs.append((user_id, emb))
            except (ValueError, TypeError) as e:
                print(f"Error processing embedding for user {user_id}: {e}")
    
    return user_embs

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- ACTION GENERATION ---

async def generate_actions(user_id: str, image_bytes: bytes, insights: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate possible actions based on insights and history"""
    system_instruction = (
        "You are an agentic assistant that proposes helpful actions in response to observations. "
        "Generate 3 distinct actions the agent could take to help the user. "
        "Each action should have: type (suggestion, question, exercise), content (what to say/ask/do), "
        "and predicted_effect (expected outcome if this action is chosen)."
    )
    
    # Format history for context
    history_context = "\n\nRecent interaction history:\n"
    for entry in history[:3]:  # Only use the most recent 3 entries
        h_timestamp = entry.get("timestamp", "Unknown time")
        h_insights = entry.get("insights", "No insights")
        h_action = entry.get("action_content", "No action")
        h_reward = entry.get("reward_value", "No reward")
        
        history_context += f"- Time: {h_timestamp}\n"
        history_context += f"  Insights: {h_insights[:100]}...\n"
        history_context += f"  Action: {h_action}\n"
        history_context += f"  Reward: {h_reward}\n\n"
    
    prompt = (
        f"{system_instruction}\n\n"
        f"Current insights: {insights}\n"
        f"{history_context}\n\n"
        "Generate 3 actions in JSON format like this:\n"
        "[\n"
        "  {\n"
        "    \"type\": \"suggestion\",\n"
        "    \"content\": \"Consider taking a short walk to refresh your mind.\",\n"
        "    \"predicted_effect\": \"May increase energy and improve mood.\"\n"
        "  },\n"
        "  {...}\n"
        "]"
    )
    
    # Generate actions using Gemini
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    
    response_text = response.text if hasattr(response, 'text') else ""
    
    # Extract JSON from response
    try:
        # Find JSON content between square brackets
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            actions = json.loads(json_str)
            return actions
        else:
            print("No valid JSON found in response")
            return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse action JSON: {e}")
        print(f"Response was: {response_text}")
        return []

# --- WORLD MODEL ---

class EnhancedWorldModel:
    """Enhanced world model that generalizes using neural networks or falls back to table lookups"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.transition_table = {}  # Fallback table-based model
        self.experiences = ExperienceReplay()
        self.state_action_pairs = {}  # Cache for embeddings
        
        # Neural network components (if available)
        self.nn_model = None
        self.state_encoder = None
        self.action_encoder = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        self.load_from_db()
        
        if TORCH_AVAILABLE:
            self._init_neural_components()
    
    def _init_neural_components(self):
        """Initialize neural network components"""
        if not TORCH_AVAILABLE:
            return
        
        # Initialize encoders
        self.state_encoder = StateActionEncoder(input_dim=EMBEDDING_DIM, output_dim=EMBEDDING_DIM).to(self.device)
        self.action_encoder = StateActionEncoder(input_dim=EMBEDDING_DIM, output_dim=EMBEDDING_DIM).to(self.device)
        
        # Initialize world model
        self.nn_model = NeuralWorldModel(
            state_dim=EMBEDDING_DIM, 
            action_dim=EMBEDDING_DIM
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.action_encoder.parameters()) + 
            list(self.nn_model.parameters()),
            lr=LEARNING_RATE
        )
        
        # Try to load pretrained weights
        self._load_model_weights()
    
    def _get_state_embedding(self, state: str) -> np.ndarray:
        """Get embedding for a state representation"""
        if state in self.state_action_pairs:
            return self.state_action_pairs[state]
        
        # Generate embedding via Gemini API
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=state[:8000]  # Limit content size
            )
            emb = np.array(response.embeddings[0].values, dtype=np.float32)
            
            # Cache the embedding
            self.state_action_pairs[state] = emb
            
            # Also save to DB for future use
            self._save_embedding("state", state, emb)
            
            return emb
        except Exception as e:
            print(f"Error generating state embedding: {e}")
            # Fallback to a random embedding
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)
    
    def _get_action_embedding(self, action: str) -> np.ndarray:
        """Get embedding for an action representation"""
        if action in self.state_action_pairs:
            return self.state_action_pairs[action]
        
        # Generate embedding via Gemini API
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=action[:8000]  # Limit content size
            )
            emb = np.array(response.embeddings[0].values, dtype=np.float32)
            
            # Cache the embedding
            self.state_action_pairs[action] = emb
            
            # Also save to DB for future use
            self._save_embedding("action", action, emb)
            
            return emb
        except Exception as e:
            print(f"Error generating action embedding: {e}")
            # Fallback to a random embedding
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)
    
    def _save_embedding(self, type_str: str, raw_data: str, embedding: np.ndarray):
        """Save embedding to database"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        timestamp = datetime.datetime.now(UTC).isoformat()
        
        c.execute(
            "INSERT INTO embeddings (user_id, type, raw_data, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
            (self.user_id, type_str, raw_data[:1000], embedding.tobytes(), timestamp)
        )
        
        conn.commit()
        conn.close()
    
    def _save_model_weights(self):
        """Save model weights to database"""
        if not TORCH_AVAILABLE or self.nn_model is None:
            return
        
        try:
            # Serialize model weights
            buffer = io.BytesIO()
            torch.save({
                'state_encoder': self.state_encoder.state_dict(),
                'action_encoder': self.action_encoder.state_dict(),
                'world_model': self.nn_model.state_dict(),
            }, buffer)
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            timestamp = datetime.datetime.now(UTC).isoformat()
            
            # Calculate accuracy metrics
            metrics = self._evaluate_model()
            
            c.execute(
                "INSERT INTO model_weights (user_id, model_name, weights, timestamp, performance_metrics) VALUES (?, ?, ?, ?, ?)",
                (self.user_id, 'world_model', buffer.getvalue(), timestamp, json.dumps(metrics))
            )
            
            conn.commit()
            conn.close()
            
            print(f"Saved world model weights. Metrics: {metrics}")
        except Exception as e:
            print(f"Error saving model weights: {e}")
    
    def _load_model_weights(self):
        """Load most recent model weights from database"""
        if not TORCH_AVAILABLE or self.nn_model is None:
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT weights FROM model_weights WHERE user_id = ? AND model_name = 'world_model' ORDER BY timestamp DESC LIMIT 1",
                (self.user_id,)
            )
            row = c.fetchone()
            conn.close()
            
            if row:
                weights_blob = row[0]
                buffer = io.BytesIO(weights_blob)
                weights = torch.load(buffer, map_location=self.device)
                
                self.state_encoder.load_state_dict(weights['state_encoder'])
                self.action_encoder.load_state_dict(weights['action_encoder'])
                self.nn_model.load_state_dict(weights['world_model'])
                
                print("Loaded world model weights from database")
                return True
        except Exception as e:
            print(f"Error loading model weights: {e}")
        
        return False
    
    def _evaluate_model(self):
        """Evaluate model on validation set"""
        if not TORCH_AVAILABLE or self.nn_model is None or len(self.experiences) < 10:
            return {"state_mse": 1.0, "reward_mse": 1.0, "samples": 0}
        
        # Sample some experiences for validation
        batch = self.experiences.sample(min(32, len(self.experiences)))
        if batch is None:
            return {"state_mse": 1.0, "reward_mse": 1.0, "samples": 0}
        
        states, actions, rewards, next_states, _ = batch
        
        # Prepare tensors
        state_tensors = []
        action_tensors = []
        next_state_tensors = []
        reward_tensors = []
        
        for s, a, r, ns, _ in zip(states, actions, rewards, next_states, _):
            with torch.no_grad():
                state_emb = torch.from_numpy(s).float().to(self.device)
                action_emb = torch.from_numpy(a).float().to(self.device)
                next_state_emb = torch.from_numpy(ns).float().to(self.device)
                reward_tensor = torch.tensor([r]).float().to(self.device)
                
                state_tensors.append(state_emb)
                action_tensors.append(action_emb)
                next_state_tensors.append(next_state_emb)
                reward_tensors.append(reward_tensor)
        
        # Stack tensors
        states_tensor = torch.stack(state_tensors)
        actions_tensor = torch.stack(action_tensors)
        target_next_states = torch.stack(next_state_tensors)
        target_rewards = torch.cat(reward_tensors)
        
        # Forward pass
        pred_next_states, pred_rewards = self.nn_model(states_tensor, actions_tensor)
        
        # Calculate losses
        state_mse = F.mse_loss(pred_next_states, target_next_states).item()
        reward_mse = F.mse_loss(pred_rewards.squeeze(), target_rewards).item()
        
        return {
            "state_mse": state_mse,
            "reward_mse": reward_mse,
            "samples": len(states)
        }
    
    def _train_model(self, batch_size=None):
        """Train neural model on experiences"""
        if not TORCH_AVAILABLE or self.nn_model is None or len(self.experiences) < batch_size:
            return
        
        bs = batch_size or BATCH_SIZE
        batch = self.experiences.sample(bs)
        if batch is None:
            return
        
        states, actions, rewards, next_states, _ = batch
        
        # Prepare tensors
        state_tensors = []
        action_tensors = []
        next_state_tensors = []
        reward_tensors = []
        
        for s, a, r, ns, _ in zip(states, actions, rewards, next_states, _):
            state_emb = torch.from_numpy(s).float().to(self.device)
            action_emb = torch.from_numpy(a).float().to(self.device)
            next_state_emb = torch.from_numpy(ns).float().to(self.device)
            reward_tensor = torch.tensor([r]).float().to(self.device)
            
            state_tensors.append(state_emb)
            action_tensors.append(action_emb)
            next_state_tensors.append(next_state_emb)
            reward_tensors.append(reward_tensor)
        
        # Stack tensors
        states_tensor = torch.stack(state_tensors)
        actions_tensor = torch.stack(action_tensors)
        target_next_states = torch.stack(next_state_tensors)
        target_rewards = torch.cat(reward_tensors)
        
        # Forward pass
        self.optimizer.zero_grad()
        pred_next_states, pred_rewards = self.nn_model(states_tensor, actions_tensor)
        
        # Calculate losses
        state_loss = F.mse_loss(pred_next_states, target_next_states)
        reward_loss = F.mse_loss(pred_rewards.squeeze(), target_rewards)
        total_loss = state_loss + reward_loss
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        # Log metrics
        signal_recorder.record_signal(
            "world_model_loss", 
            total_loss.item(),
            {
                "state_loss": state_loss.item(),
                "reward_loss": reward_loss.item(),
                "batch_size": bs
            }
        )
        
        return total_loss.item()
    
    def load_from_db(self):
        """Load model data from database"""
        # Load transition table (fallback model)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT state_before, predicted_action, actual_next_state, actual_reward
            FROM predictions 
            WHERE actual_next_state IS NOT NULL AND actual_reward IS NOT NULL
        """)
        rows = c.fetchall()
        
        # Also load embeddings for previously seen states/actions
        c.execute("SELECT type, raw_data, embedding FROM embeddings WHERE user_id = ?", (self.user_id,))
        embedding_rows = c.fetchall()
        
        conn.close()
        
        # Process transition data
        for state, action, next_state, reward in rows:
            key = (state, action)
            if key not in self.transition_table:
                self.transition_table[key] = ([], [])
            
            states, rewards = self.transition_table[key]
            states.append(next_state)
            rewards.append(reward)
            
            # Also add to experience replay
            try:
                state_emb = self._get_state_embedding(state)
                action_emb = self._get_action_embedding(action)
                next_state_emb = self._get_state_embedding(next_state)
                
                self.experiences.add(
                    state_emb, 
                    action_emb, 
                    reward, 
                    next_state_emb,
                    False  # done flag
                )
            except Exception as e:
                print(f"Error adding experience: {e}")
        
        # Process embedding data
        for type_str, raw_data, emb_blob in embedding_rows:
            try:
                embedding = np.frombuffer(emb_blob, dtype=np.float32)
                self.state_action_pairs[raw_data] = embedding
            except Exception as e:
                print(f"Error loading embedding: {e}")
    
    def predict_outcome(self, current_state: str, action: str) -> Tuple[str, float]:
        """Predict the next state and reward for taking an action"""
        # First try neural prediction if available
        if TORCH_AVAILABLE and self.nn_model is not None and len(self.experiences) > 10:
            try:
                # Get embeddings
                state_emb = self._get_state_embedding(current_state)
                action_emb = self._get_action_embedding(action)
                
                # Convert to tensors
                state_tensor = torch.from_numpy(state_emb).float().unsqueeze(0).to(self.device)
                action_tensor = torch.from_numpy(action_emb).float().unsqueeze(0).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    next_state_emb, reward = self.nn_model(state_tensor, action_tensor)
                
                # Find closest state for next_state_emb
                next_state_emb_np = next_state_emb.squeeze().cpu().numpy()
                closest_state = current_state  # Default to same state
                best_sim = -1.0
                
                # Find closest matching state based on embedding similarity
                for s in self.state_action_pairs:
                    if not s.startswith('{'): continue  # Skip non-state entries
                    s_emb = self.state_action_pairs[s]
                    sim = cosine_similarity(next_state_emb_np, s_emb)
                    if sim > best_sim:
                        best_sim = sim
                        closest_state = s
                
                # Return prediction
                return closest_state, float(reward.item())
            
            except Exception as e:
                print(f"Error in neural prediction: {e}")
                # Fall back to table lookup
        
        # Fallback to simple table lookup
        key = (current_state, action)
        if key in self.transition_table and len(self.transition_table[key][0]) > 0:
            states, rewards = self.transition_table[key]
            # Return the most recent observed state and average reward
            return states[-1], sum(rewards) / len(rewards)
        
        # If we have no data, default to predicting no change and neutral reward
        return current_state, 0.0
    
    def update(self, state_before: str, action: str, next_state: str, reward: float):
        """Update the model with a new observation"""
        # Update table-based model
        key = (state_before, action)
        if key not in self.transition_table:
            self.transition_table[key] = ([], [])
        
        states, rewards = self.transition_table[key]
        states.append(next_state)
        rewards.append(reward)
        
        # Save to database
        save_prediction(state_before, action, states[0], 
                       rewards[0] if rewards else 0.0, 
                       next_state, reward)
        
        # Update neural model
        if TORCH_AVAILABLE and self.nn_model is not None:
            try:
                # Get embeddings
                state_emb = self._get_state_embedding(state_before)
                action_emb = self._get_action_embedding(action)
                next_state_emb = self._get_state_embedding(next_state)
                
                # Add to experience replay
                self.experiences.add(
                    state_emb, 
                    action_emb, 
                    reward, 
                    next_state_emb,
                    False  # done flag
                )
                
                # Train model if we have enough experiences
                if len(self.experiences) >= BATCH_SIZE:
                    self._train_model()
                
                # Periodically save model weights
                if random.random() < 0.05:  # ~5% chance to save model
                    self._save_model_weights()
                
            except Exception as e:
                print(f"Error updating neural model: {e}")

class DQNAgent:
    """Deep Q-Network agent for learning optimal action selection"""
    
    def __init__(self, user_id: str, state_dim=EMBEDDING_DIM, action_dim=EMBEDDING_DIM):
        self.user_id = user_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.experiences = ExperienceReplay()
        self.training_iterations = 0
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        
        # Initialize networks if torch is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        if TORCH_AVAILABLE:
            self._init_networks()
        
        # Load previous experiences
        self.load_from_db()
    
    def _init_networks(self):
        """Initialize Q-networks"""
        if not TORCH_AVAILABLE:
            return
        
        # Create networks
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        
        # Copy weights from Q to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Try to load saved weights
        self._load_weights()
    
    def _save_weights(self):
        """Save model weights to database"""
        if not TORCH_AVAILABLE or self.q_network is None:
            return
        
        try:
            # Serialize weights
            buffer = io.BytesIO()
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_iterations': self.training_iterations
            }, buffer)
            
            # Calculate performance metrics
            metrics = self._evaluate()
            
            # Save to database
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            timestamp = datetime.datetime.now(UTC).isoformat()
            
            c.execute(
                "INSERT INTO model_weights (user_id, model_name, weights, timestamp, performance_metrics) VALUES (?, ?, ?, ?, ?)",
                (self.user_id, 'q_network', buffer.getvalue(), timestamp, json.dumps(metrics))
            )
            
            conn.commit()
            conn.close()
            
            print(f"Saved Q-network weights. Metrics: {metrics}")
            return True
            
        except Exception as e:
            print(f"Error saving Q-network weights: {e}")
            return False
    
    def _load_weights(self):
        """Load model weights from database"""
        if not TORCH_AVAILABLE or self.q_network is None:
            return
        
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT weights FROM model_weights WHERE user_id = ? AND model_name = 'q_network' ORDER BY timestamp DESC LIMIT 1",
                (self.user_id,)
            )
            row = c.fetchone()
            conn.close()
            
            if row:
                weights_blob = row[0]
                buffer = io.BytesIO(weights_blob)
                checkpoint = torch.load(buffer, map_location=self.device)
                
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', EPSILON_START)
                self.training_iterations = checkpoint.get('training_iterations', 0)
                
                print("Loaded Q-network weights from database")
                return True
            
        except Exception as e:
            print(f"Error loading Q-network weights: {e}")
        
        return False
    
    def load_from_db(self):
        """Load experiences from database predictions"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Load state and action embeddings
        c.execute("SELECT type, raw_data, embedding FROM embeddings WHERE user_id = ?", (self.user_id,))
        embedding_rows = c.fetchall()
        
        # Create dictionaries for embeddings
        state_embeddings = {}
        action_embeddings = {}
        
        for type_str, raw_data, emb_blob in embedding_rows:
            try:
                embedding = np.frombuffer(emb_blob, dtype=np.float32)
                if type_str == 'state':
                    state_embeddings[raw_data] = embedding
                elif type_str == 'action':
                    action_embeddings[raw_data] = embedding
            except Exception as e:
                print(f"Error loading embedding: {e}")
        
        # Load predictions with actual outcomes for RL training
        c.execute("""
            SELECT p.state_before, p.predicted_action, p.actual_reward, p.actual_next_state
            FROM predictions p
            WHERE p.actual_next_state IS NOT NULL AND p.actual_reward IS NOT NULL
            ORDER BY p.timestamp
        """)
        rows = c.fetchall()
        conn.close()
        
        # Add experiences to replay buffer
        for state, action, reward, next_state in rows:
            if state in state_embeddings and action in action_embeddings and next_state in state_embeddings:
                try:
                    state_emb = state_embeddings[state]
                    action_emb = action_embeddings[action]
                    next_state_emb = state_embeddings[next_state]
                    
                    self.experiences.add(
                        state_emb,
                        action_emb,
                        reward,
                        next_state_emb,
                        False  # done flag (we don't have episode termination in this setup)
                    )
                except Exception as e:
                    print(f"Error adding experience to DQN buffer: {e}")
    
    def select_action(self, state_emb: np.ndarray, possible_actions: List[Dict[str, Any]], world_model) -> int:
        """Select action based on Q-values and exploration policy"""
        if not TORCH_AVAILABLE or self.q_network is None or len(possible_actions) == 0:
            # Fallback to random selection without torch
            return random.randint(0, len(possible_actions) - 1)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, len(possible_actions) - 1)
        
        # Exploitation: select highest value action
        try:
            state_tensor = torch.from_numpy(state_emb).float().unsqueeze(0).to(self.device)
            
            q_values = []
            for action in possible_actions:
                action_json = json.dumps(action)
                try:
                    # Get action embedding from world model (which caches embeddings)
                    action_emb = world_model._get_action_embedding(action_json)
                    action_tensor = torch.from_numpy(action_emb).float().unsqueeze(0).to(self.device)
                    
                    # Calculate Q-value
                    with torch.no_grad():
                        q = self.q_network(state_tensor, action_tensor)
                    
                    q_values.append(q.item())
                    
                except Exception as e:
                    print(f"Error computing Q-value: {e}")
                    q_values.append(0.0)  # Default to zero
            
            # Return index of action with highest Q-value
            return q_values.index(max(q_values))
            
        except Exception as e:
            print(f"Error in DQN action selection: {e}")
            # Fallback to random selection on error
            return random.randint(0, len(possible_actions) - 1)
    
    def update(self, state_emb, action_emb, reward, next_state_emb, done=False):
        """Update Q-network with new experience"""
        # Add to replay buffer
        self.experiences.add(state_emb, action_emb, reward, next_state_emb, done)
        
        # Only train if we have PyTorch and enough samples
        if not TORCH_AVAILABLE or self.q_network is None or len(self.experiences) < BATCH_SIZE:
            return
        
        # Sample batch
        batch = self.experiences.sample(BATCH_SIZE)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_tensor, actions_tensor).squeeze()
        
        # Compute target Q-values using target network
        # For each next state, we compute the Q-value for all possible actions and take the max
        # Since we don't have explicit action choices for next states, we'll use the actions we have
        # This is a simplification - in practice, you'd want to sample or enumerate possible next actions
        target_q_values = rewards_tensor.clone()
        
        # Calculate target values only for non-terminal states
        non_terminal_mask = 1.0 - dones_tensor
        next_q_values = torch.zeros_like(rewards_tensor)
        
        # Find max Q-value across all actions for each next state
        # This is computationally expensive but more accurate
        for i in range(len(next_states)):
            next_state = next_states_tensor[i:i+1]
            max_q = float('-inf')
            
            # Use the same actions for each state - not ideal but practical
            for j in range(len(actions)):
                action = actions_tensor[j:j+1]
                with torch.no_grad():
                    q = self.target_network(next_state, action).item()
                max_q = max(max_q, q)
            
            next_q_values[i] = max_q
        
        # Q-Learning update: Q(s,a) = r + γ * max_a' Q(s',a')
        target_q_values = rewards_tensor + GAMMA * next_q_values * non_terminal_mask
        
        # Compute loss and update
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q_values, target_q_values.detach())
        loss.backward()
        self.optimizer.step()
        
        # Log training metrics
        signal_recorder.record_signal(
            "dqn_loss", 
            loss.item(),
            {"batch_size": BATCH_SIZE}
        )
        
        # Update target network periodically
        self.training_iterations += 1
        if self.training_iterations % TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Periodically save model
            if self.training_iterations % (TARGET_UPDATE_FREQ * 5) == 0:
                self._save_weights()
        
        # Decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
        # Record epsilon value
        signal_recorder.record_signal("epsilon", self.epsilon)
        
        return loss.item()
    
    def _evaluate(self):
        """Evaluate agent performance on validation set"""
        if not TORCH_AVAILABLE or self.q_network is None or len(self.experiences) < 10:
            return {"q_mse": 1.0, "samples": 0, "epsilon": self.epsilon}
        
        # Sample a validation batch
        batch = self.experiences.sample(min(32, len(self.experiences)))
        if batch is None:
            return {"q_mse": 1.0, "samples": 0, "epsilon": self.epsilon}
        
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        
        # Compute predicted Q-values
        with torch.no_grad():
            predicted_q = self.q_network(states_tensor, actions_tensor).squeeze()
            
            # Use rewards as simple estimate of "true" Q-values for evaluation
            # This is a rough approximation
            q_mse = F.mse_loss(predicted_q, rewards_tensor).item()
        
        return {
            "q_mse": q_mse,
            "samples": len(states),
            "epsilon": self.epsilon,
            "iterations": self.training_iterations
        }

# --- REWARD FUNCTIONS ---

async def get_user_feedback(action_content: str) -> Tuple[float, str]:
    """Get explicit feedback from user on the action (placeholder for GUI implementation)"""
    print(f"\nAction: {action_content}")
    print("How helpful was this? (1-5, 5 being very helpful): ")
    
    # Use asyncio to handle input without blocking
    loop = asyncio.get_event_loop()
    try:
        # Wait for input with a timeout
        user_input = await asyncio.wait_for(
            loop.run_in_executor(None, input), 
            timeout=10.0
        )
        
        try:
            rating = float(user_input.strip())
            if rating < 1 or rating > 5:
                print(f"Invalid rating: {rating}. Using default value 3.")
                rating = 3.0
        except ValueError:
            print(f"Invalid input format. Using default value 3.")
            rating = 3.0
            
        # Normalize to [-1, 1] range
        normalized_rating = (rating - 3) / 2
        return normalized_rating, "explicit_feedback"
    
    except asyncio.TimeoutError:
        print("Input timeout. Using neutral feedback.")
        return 0.0, "timeout_neutral"
    except Exception as e:
        print(f"Error getting feedback: {e}. Using neutral value.")
        return 0.0, "error_neutral"

async def estimate_mood_from_image(image_bytes: bytes) -> float:
    """Estimate user's mood from facial expression using Gemini"""
    prompt = (
        "Analyze the person's facial expression and estimate their mood on a scale from -1 to 1, where:\n"
        "-1 = very negative mood (distressed, angry, sad)\n"
        "0 = neutral mood\n"
        "1 = very positive mood (happy, excited, content)\n\n"
        "Respond with ONLY a single number between -1 and 1."
    )
    
    user_parts = [
        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)),
        types.Part(text=prompt)
    ]
    
    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=user_parts
    )
    
    response_text = response.text.strip() if hasattr(response, 'text') else "0.0"
    
    try:
        # Extract the number from the response
        mood_value = float(response_text)
        # Ensure it's in the range [-1, 1]
        mood_value = max(-1.0, min(1.0, mood_value))
        return mood_value
    except ValueError:
        print(f"Failed to parse mood value from: '{response_text}'")
        return 0.0  # Default to neutral

def calculate_composite_reward(mood_value: float, user_feedback: float, action_novelty: float, 
                              preferences: Dict[str, float]) -> float:
    """Calculate composite reward based on multiple signals and user preferences"""
    # Combine signals based on user's preferences
    composite_reward = (
        preferences.get("mood_improvement", 1.0) * mood_value +
        preferences.get("engagement", 0.8) * user_feedback +
        preferences.get("novelty", 0.5) * action_novelty
    )
    
    # Normalize to be within [-1, 1]
    weight_sum = sum(preferences.values())
    if weight_sum > 0:
        composite_reward /= weight_sum
    
    return composite_reward

# --- USER IDENTIFICATION ---

async def identify_user(image_bytes: bytes) -> str:
    """Identify user from face embedding with improved confidence handling"""
    new_desc = await generate_face_description(image_bytes)
    new_emb = get_face_embedding(new_desc)
    
    # Make sure database is initialized
    init_db()
    
    user_embs = get_all_user_embeddings()
    
    if not user_embs:
        print("No existing users found. Creating new user profile.")
        user_id = await create_new_user()
        # Save the embedding for this new user
        save_user_embedding(user_id, new_emb)
        return user_id
    
    scored_candidates = []
    for user_id, avg_emb in user_embs:
        score = cosine_similarity(new_emb, avg_emb)
        scored_candidates.append((user_id, score))
    
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top3 = scored_candidates[:3]
    
    print("Top 3 most likely users (embedding):")
    for idx, (user_id, score) in enumerate(top3, 1):
        print(f"  {idx}. user_id: {user_id} (confidence: {score:.2f})")
    
    if top3 and top3[0][1] > 0.50:  # Increased confidence threshold
        best_user = top3[0][0]
        print(f"Identified user '{best_user}' with confidence {top3[0][1]:.2f}.")
        return best_user
    else:
        print("No confident match found.")
        if len(top3) > 0 and top3[0][1] > 0.7:
            print(f"Did you mean '{top3[0][0]}'? [y/N]")
            resp = input().strip().lower()
            if resp == 'y':
                return top3[0][0]
        
        # Create new user if no match
        user_id = await create_new_user()
        # Save the embedding for this new user
        save_user_embedding(user_id, new_emb)
        return user_id

async def create_new_user() -> str:
    """Create a new user profile with preferences"""
    print("Enter a new user_id:")
    user_id = input().strip()
    
    # Set default preferences
    preferences = {
        "mood_improvement": 1.0,
        "engagement": 0.8,
        "novelty": 0.5
    }
    
    # Ask if user wants to customize preferences
    print("Do you want to customize reward preferences? [y/N]")
    resp = input().strip().lower()
    if resp == 'y':
        for pref_type in preferences.keys():
            print(f"Enter weight for {pref_type} (0.0-1.0, default: {preferences[pref_type]}):")
            try:
                value = float(input().strip())
                if 0.0 <= value <= 1.0:
                    preferences[pref_type] = value
            except ValueError:
                print(f"Invalid input, using default: {preferences[pref_type]}")
    
    # Save preferences
    set_user_reward_preferences(user_id, preferences)
    
    return user_id

# --- MAIN PIPELINE ---

async def perform_interaction_cycle(user_id: str, agent_state: AgentState, world_model: EnhancedWorldModel, dqn_agent: DQNAgent, experiment_tracker: ExperimentTracker):
    """Perform a single perception-action-reward cycle"""
    print("\n--- Starting interaction cycle ---")
    
    # Step 1: Capture image
    print("Capturing image...")
    try:
        image_bytes = capture_image()
        print("Image captured.")
    except RuntimeError as e:
        print(f"Error capturing image: {e}")
        return
    
    # Step 2: Get historical context
    last_insight = get_last_insight(user_id)
    history = get_historical_interactions(user_id)
    
    # Step 3: Extract insights
    print("Extracting insights...")
    insights = await extract_insights(image_bytes, last_insight, history)
    print("Insights extracted.")
    
    # Step 4: Update agent state with new observation
    face_description = await generate_face_description(image_bytes)
    face_embedding = get_face_embedding(face_description)
    observation_id = save_observation(user_id, image_bytes, insights, agent_state.to_json(), face_description, face_embedding)
    save_user_embedding(user_id, face_embedding)
    
    agent_state.update(new_observation={"id": observation_id, "insights": insights})
    
    # Get state embedding for RL
    state_emb = world_model._get_state_embedding(agent_state.to_json())
    
    # Step 5: Generate possible actions
    print("Generating actions...")
    actions = await generate_actions(user_id, image_bytes, insights, history)
    
    if not actions:
        print("No valid actions generated.")
        return
    
    # Step 6: Select action using DQN policy instead of greedy world model selection
    selected_idx = dqn_agent.select_action(state_emb, actions, world_model)
    best_action = actions[selected_idx]
    
    print(f"Selected action: {best_action['type']} - {best_action['content']}")
    action_content = best_action['content']
    action_id = save_action(observation_id, best_action['type'], action_content)
    action_json = json.dumps(best_action)
    action_emb = world_model._get_action_embedding(action_json)
    agent_state.update(new_action=best_action)
    
    # Step 7: Execute action in the real world if applicable
    spotify_played = False
    if best_action['type'] == 'focus_music' and spotify_actuator and spotify_actuator.available:
        spotify_played = spotify_actuator.play_focus_playlist()
        if spotify_played:
            print("Executing real-world action: Playing focus music on Spotify")
            spotify_actuator.save_playlist_action(user_id, action_id)
    
    # Step 8: Collect rewards
    # 8.1: Get mood from image
    print("Estimating mood...")
    mood_value = await estimate_mood_from_image(image_bytes)
    agent_state.update(new_mood=mood_value)
    
    # 8.2: Get explicit feedback from user
    print("Getting user feedback...")
    user_feedback, feedback_source = await get_user_feedback(action_content)
    
    # 8.3: Calculate action novelty (higher reward for novel actions)
    # Simple implementation: check if this action type has been used before
    action_type = best_action['type']
    action_novelty = 0.0
    if action_type not in agent_state.successful_strategies:
        action_novelty = 0.5  # Moderate bonus for new action types
    
    # 8.4: Get user reward preferences
    preferences = get_user_reward_preferences(user_id)
    
    # 8.5: Calculate composite reward
    composite_reward = calculate_composite_reward(
        mood_value, user_feedback, action_novelty, preferences
    )
    
    # 8.6: Add Spotify feedback if applicable
    if spotify_played and spotify_actuator:
        # Check if user skipped the track (negative signal)
        skipped = spotify_actuator.is_track_skipped()
        if skipped:
            # Apply penalty for skipped music
            composite_reward -= 0.3
            spotify_actuator.save_playlist_action(user_id, action_id, skipped=True)
    
    # 8.7: Save reward to database
    context = json.dumps({
        "mood": mood_value,
        "explicit_feedback": user_feedback,
        "novelty": action_novelty,
        "spotify_played": spotify_played
    })
    save_reward(action_id, composite_reward, "composite", context)
    
    # 8.8: Update agent state with reward
    agent_state.update(new_reward=composite_reward)
    
    # Step 9: Update models
    
    # 9.1: Update world model
    world_model.update(
        agent_state.to_json(),
        json.dumps(best_action),
        agent_state.to_json(),  # Updated state after action
        composite_reward
    )
    
    # 9.2: Update DQN agent
    next_state_emb = world_model._get_state_embedding(agent_state.to_json())
    dqn_agent.update(state_emb, action_emb, composite_reward, next_state_emb)
    
    # 9.3: Log to experiment tracker if running an experiment
    if experiment_tracker and experiment_tracker.current_experiment:
        experiment_tracker.log_reward(composite_reward, action_type, user_feedback)
    
    # 9.4: Log metrics
    signal_recorder.record_signal(
        "reward", 
        composite_reward,
        {
            "action_type": action_type,
            "mood": mood_value,
            "user_feedback": user_feedback,
            "novelty": action_novelty
        }
    )
    
    print(f"Cycle completed. Composite reward: {composite_reward:.2f}")

async def agentic_insight_loop():
    """Main continuous agent loop that persists across sessions"""
    init_db()
    
    print("Starting agentic insight loop...")
    
    # Initial image capture and user identification
    print("Capturing initial image for user identification...")
    try:
        image_bytes = capture_image()
        print("Image captured.")
    except RuntimeError as e:
        print(f"Error capturing image: {e}")
        return

    # Identify user
    print("Identifying user...")
    user_id = await identify_user(image_bytes)
    print(f"User identified: {user_id}")
    
    # Initialize components
    agent_state = AgentState(user_id)
    world_model = EnhancedWorldModel(user_id)
    dqn_agent = DQNAgent(user_id)
    experiment_tracker = ExperimentTracker(user_id)
    
    # Display welcome message
    print(f"\nWelcome, {user_id}! I'll be observing and learning from our interactions.")
    
    # Ask if user wants to run an experiment
    print("\nWould you like to run an experiment? [y/N]")
    run_experiment = input().strip().lower() == 'y'
    
    if run_experiment:
        print("Select experiment type:")
        print("1. Neural World Model vs. Table Lookup (Random baseline)")
        print("2. DQN Action Selection vs. Random Selection")
        print("3. Ablation Study (all components)")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            experiment_tracker.start_experiment("neural_vs_table", {
                "description": "Comparing neural world model to table lookup",
                "world_model": "neural" if TORCH_AVAILABLE else "table",
                "policy": "dqn" if TORCH_AVAILABLE else "random"
            })
        elif choice == '2':
            experiment_tracker.start_experiment("dqn_vs_random", {
                "description": "Comparing DQN action selection to random selection",
                "world_model": "neural" if TORCH_AVAILABLE else "table",
                "policy": "dqn" if TORCH_AVAILABLE else "random" 
            })
        elif choice == '3':
            experiment_tracker.start_experiment("ablation_study", {
                "description": "Full ablation study of all components",
                "world_model": "neural" if TORCH_AVAILABLE else "table",
                "policy": "dqn" if TORCH_AVAILABLE else "random",
                "spotify": spotify_actuator.available if spotify_actuator else False
            })
    
    print("Press Ctrl+C at any time to exit.")
    
    try:
        cycle_count = 0
        while True:
            # Perform a single interaction cycle
            await perform_interaction_cycle(user_id, agent_state, world_model, dqn_agent, experiment_tracker)
            cycle_count += 1
            
            # Optionally re-identify user periodically (every 5 cycles)
            if cycle_count % 5 == 0:
                print("\nVerifying user identity...")
                try:
                    image_bytes = capture_image()
                    verified_user_id = await identify_user(image_bytes)
                    if verified_user_id != user_id:
                        print(f"User changed from {user_id} to {verified_user_id}")
                        user_id = verified_user_id
                        agent_state = AgentState(user_id)
                        world_model = EnhancedWorldModel(user_id)
                        dqn_agent = DQNAgent(user_id)
                        
                        # End any running experiments
                        if experiment_tracker.current_experiment:
                            experiment_tracker.end_experiment()
                        experiment_tracker = ExperimentTracker(user_id)
                except RuntimeError as e:
                    print(f"Error during user verification: {e}")
            
            # Wait between cycles
            print("\nWaiting for next cycle...")
            await asyncio.sleep(CYCLE_WAIT_TIME)
    
    except KeyboardInterrupt:
        print("\nExiting agentic insight loop.")
        
        # End any running experiments
        if experiment_tracker.current_experiment:
            results = experiment_tracker.end_experiment()
            
            # Show visualization if we have data
            if MATPLOTLIB_AVAILABLE and len(experiment_tracker.experiment_data) > 0:
                print("Generating experiment visualization...")
                experiment_tracker.visualize_experiment_results()
                
    except Exception as e:
        print(f"\nError in agentic insight loop: {e}")
        import traceback
        traceback.print_exc()

# --- ENTRY POINT ---

if __name__ == "__main__":
    asyncio.run(agentic_insight_loop())

class SpotifyActuator:
    """Spotify actuator for playing focus playlists as a real-world action"""
    
    def __init__(self):
        self.sp = None
        self.available = SPOTIFY_AVAILABLE and all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET])
        self.authenticated = False
        self.focus_playlists = []
        self.active_playlist = None
        self.active_playlist_name = None
        
        if self.available:
            self._initialize()
    
    def _initialize(self):
        """Initialize Spotify client"""
        try:
            scope = "user-read-playback-state,user-modify-playback-state,user-read-currently-playing"
            
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope,
                open_browser=False
            ))
            
            # Test authentication
            self.sp.current_user()
            self.authenticated = True
            
            # Cache focus playlists
            self._cache_focus_playlists()
            
            print("Spotify integration initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing Spotify: {e}")
            self.authenticated = False
            return False
    
    def _cache_focus_playlists(self):
        """Cache focus and productivity playlists from Spotify"""
        if not self.authenticated:
            return []
        
        try:
            # Search for focus playlists
            focus_results = self.sp.search(q='focus', type='playlist', limit=5)
            productivity_results = self.sp.search(q='productivity', type='playlist', limit=5)
            study_results = self.sp.search(q='study', type='playlist', limit=5)
            
            playlists = []
            
            # Process focus playlists
            for playlist in focus_results['playlists']['items']:
                playlists.append({
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist.get('description', ''),
                    'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                    'category': 'focus'
                })
            
            # Process productivity playlists
            for playlist in productivity_results['playlists']['items']:
                playlists.append({
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist.get('description', ''),
                    'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                    'category': 'productivity'
                })
            
            # Process study playlists
            for playlist in study_results['playlists']['items']:
                playlists.append({
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist.get('description', ''),
                    'image_url': playlist['images'][0]['url'] if playlist['images'] else None,
                    'category': 'study'
                })
            
            self.focus_playlists = playlists
            return playlists
            
        except Exception as e:
            print(f"Error caching Spotify playlists: {e}")
            return []
    
    def play_focus_playlist(self, playlist_id=None, playlist_category=None) -> bool:
        """Play a focus playlist based on ID or category"""
        if not self.authenticated:
            print("Spotify not authenticated")
            return False
        
        try:
            # If no playlist ID is provided, select one based on category or random
            if playlist_id is None:
                if playlist_category:
                    # Filter by category
                    matching_playlists = [p for p in self.focus_playlists if p['category'] == playlist_category]
                    if matching_playlists:
                        playlist_id = random.choice(matching_playlists)['id']
                    else:
                        # If no matching playlists, pick random
                        playlist_id = random.choice(self.focus_playlists)['id'] if self.focus_playlists else None
                else:
                    # Pick random playlist
                    playlist_id = random.choice(self.focus_playlists)['id'] if self.focus_playlists else None
            
            if not playlist_id:
                print("No playlist available to play")
                return False
            
            # Get playlist name for logging
            playlist_name = next((p['name'] for p in self.focus_playlists if p['id'] == playlist_id), "Unknown")
            
            # Play the playlist
            self.sp.start_playback(context_uri=f"spotify:playlist:{playlist_id}")
            
            self.active_playlist = playlist_id
            self.active_playlist_name = playlist_name
            
            print(f"Playing Spotify playlist: {playlist_name}")
            return True
            
        except Exception as e:
            print(f"Error playing Spotify playlist: {e}")
            return False
    
    def stop_playback(self) -> bool:
        """Stop current playback"""
        if not self.authenticated:
            return False
        
        try:
            self.sp.pause_playback()
            print("Stopped Spotify playback")
            return True
        except Exception as e:
            print(f"Error stopping Spotify playback: {e}")
            return False
    
    def is_track_skipped(self) -> bool:
        """Check if user has manually skipped track since playback started"""
        if not self.authenticated or not self.active_playlist:
            return False
        
        try:
            # Get currently playing track
            current = self.sp.current_playback()
            
            # Check if user is still playing from the same playlist
            if current and current.get('context') and current['context'].get('uri'):
                current_playlist_uri = current['context']['uri']
                current_playlist_id = current_playlist_uri.split(':')[-1]
                
                return current_playlist_id != self.active_playlist
                
        except Exception as e:
            print(f"Error checking if track skipped: {e}")
        
        return False
    
    def save_playlist_action(self, user_id: str, action_id: int, skipped: bool = False):
        """Save playlist action to database"""
        if not self.active_playlist:
            return
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        timestamp = datetime.datetime.now(UTC).isoformat()
        
        c.execute(
            "INSERT INTO spotify_history (user_id, action_id, playlist_id, playlist_name, timestamp, user_skipped) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, action_id, self.active_playlist, self.active_playlist_name, timestamp, skipped)
        )
        
        conn.commit()
        conn.close()

# --- EXPERIMENT TRACKING ---

class ExperimentTracker:
    """Track and analyze experiments to evaluate agent performance"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_experiment = None
        self.experiment_data = []
        self.baseline_data = []
    
    def start_experiment(self, name: str, config: Dict[str, Any] = None):
        """Start a new experiment"""
        if self.current_experiment:
            self.end_experiment()
        
        self.current_experiment = {
            'name': name,
            'config': config or {},
            'start_time': datetime.datetime.now(UTC).isoformat(),
            'rewards': [],
            'actions': [],
            'user_feedback': []
        }
        
        print(f"Started experiment: {name}")
        
        # Record start in DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute(
            "INSERT INTO experiments (user_id, experiment_name, start_time, config) VALUES (?, ?, ?, ?)",
            (self.user_id, name, self.current_experiment['start_time'], json.dumps(config))
        )
        
        conn.commit()
        conn.close()
    
    def log_reward(self, reward: float, action_type: str, feedback: float):
        """Log reward for current experiment"""
        if not self.current_experiment:
            return
        
        self.current_experiment['rewards'].append(reward)
        self.current_experiment['actions'].append(action_type)
        self.current_experiment['user_feedback'].append(feedback)
        
        # Log to signal_recorder for real-time graphing
        signal_recorder.record_signal(
            f"experiment_{self.current_experiment['name']}_reward", 
            reward,
            {
                'action_type': action_type,
                'user_feedback': feedback,
                'experiment': self.current_experiment['name']
            }
        )
    
    def end_experiment(self):
        """End current experiment and save results"""
        if not self.current_experiment:
            return
        
        end_time = datetime.datetime.now(UTC).isoformat()
        self.current_experiment['end_time'] = end_time
        
        # Calculate results
        results = {
            'avg_reward': sum(self.current_experiment['rewards']) / max(1, len(self.current_experiment['rewards'])),
            'reward_count': len(self.current_experiment['rewards']),
            'avg_feedback': sum(self.current_experiment['user_feedback']) / max(1, len(self.current_experiment['user_feedback'])),
            'action_distribution': self._count_actions(self.current_experiment['actions'])
        }
        
        self.experiment_data.append(self.current_experiment)
        
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute(
            "UPDATE experiments SET end_time = ?, results = ? WHERE user_id = ? AND experiment_name = ? AND start_time = ?",
            (end_time, json.dumps(results), self.user_id, self.current_experiment['name'], self.current_experiment['start_time'])
        )
        
        conn.commit()
        conn.close()
        
        print(f"Ended experiment: {self.current_experiment['name']}")
        print(f"Results: {results}")
        
        self.current_experiment = None
        return results
    
    def _count_actions(self, actions: List[str]) -> Dict[str, int]:
        """Count frequency of each action type"""
        counts = {}
        for action in actions:
            if action not in counts:
                counts[action] = 0
            counts[action] += 1
        return counts
    
    def visualize_experiment_results(self, experiment_name=None):
        """Visualize experiment results"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for visualization")
            return
        
        # If no experiment name provided, use all experiments
        if experiment_name:
            experiments = [e for e in self.experiment_data if e['name'] == experiment_name]
        else:
            experiments = self.experiment_data
        
        if not experiments:
            print("No experiment data to visualize")
            return
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot rewards over time
        for exp in experiments:
            rewards = exp['rewards']
            name = exp['name']
            ax1.plot(range(len(rewards)), rewards, label=name)
        
        ax1.set_title('Rewards Over Time')
        ax1.set_xlabel('Interaction Cycle')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Plot action distribution
        exp_names = [e['name'] for e in experiments]
        action_types = set()
        for exp in experiments:
            action_types.update(exp['actions'])
        
        action_counts = []
        for exp in experiments:
            counts = self._count_actions(exp['actions'])
            action_counts.append([counts.get(action, 0) for action in action_types])
        
        x = range(len(action_types))
        width = 0.8 / len(experiments)
        
        for i, (counts, name) in enumerate(zip(action_counts, exp_names)):
            ax2.bar([p + i * width for p in x], counts, width, label=name)
        
        ax2.set_title('Action Distribution')
        ax2.set_xlabel('Action Type')
        ax2.set_ylabel('Count')
        ax2.set_xticks([p + width * (len(experiments) - 1) / 2 for p in x])
        ax2.set_xticklabels(list(action_types))
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'experiment_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()

# Create Spotify actuator instance
spotify_actuator = SpotifyActuator() if SPOTIFY_AVAILABLE else None 