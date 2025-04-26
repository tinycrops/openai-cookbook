import os
import io
import base64
import asyncio
import datetime
from datetime import UTC  # Use timezone-aware objects
import sqlite3
import json
from typing import Optional, Dict, Any, List, Tuple

import cv2
import PIL.Image
from google import genai
from google.genai import types
import numpy as np

# --- CONFIGURATION ---

DB_PATH = "agentic_insights.db"
MODEL = "models/gemini-2.0-flash"  # Fixed model name for live API
EMBED_MODEL = "text-embedding-004"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CYCLE_WAIT_TIME = 15  # Seconds to wait between interaction cycles

client = genai.Client(api_key=GEMINI_API_KEY)

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

class SimpleWorldModel:
    """A simple predictive model for user state and rewards"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        # Maps (state_representation, action_type) to (next_state_distribution, reward_distribution)
        self.transition_table = {}
        self.load_from_db()
    
    def load_from_db(self):
        """Load model data from database"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT state_before, predicted_action, actual_next_state, actual_reward
            FROM predictions 
            WHERE actual_next_state IS NOT NULL AND actual_reward IS NOT NULL
        """)
        rows = c.fetchall()
        conn.close()
        
        for state, action, next_state, reward in rows:
            key = (state, action)
            if key not in self.transition_table:
                self.transition_table[key] = ([], [])
            
            states, rewards = self.transition_table[key]
            states.append(next_state)
            rewards.append(reward)
    
    def predict_outcome(self, current_state: str, action: str) -> Tuple[str, float]:
        """Predict the next state and reward for taking an action"""
        key = (current_state, action)
        
        if key in self.transition_table and len(self.transition_table[key][0]) > 0:
            states, rewards = self.transition_table[key]
            # Simple model: just return the most recent observed state and average reward
            return states[-1], sum(rewards) / len(rewards)
        
        # If we have no data, default to predicting no change and neutral reward
        return current_state, 0.0
    
    def update(self, state_before: str, action: str, next_state: str, reward: float):
        """Update the model with a new observation"""
        key = (state_before, action)
        if key not in self.transition_table:
            self.transition_table[key] = ([], [])
        
        states, rewards = self.transition_table[key]
        states.append(next_state)
        rewards.append(reward)
        
        # Also save to database
        save_prediction(state_before, action, states[0], 
                       rewards[0] if rewards else 0.0, 
                       next_state, reward)

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

async def perform_interaction_cycle(user_id: str, agent_state: AgentState, world_model: SimpleWorldModel):
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
    
    # Step 5: Generate possible actions
    print("Generating actions...")
    actions = await generate_actions(user_id, image_bytes, insights, history)
    
    if not actions:
        print("No valid actions generated.")
        return
    
    # Step 6: Use world model to select best action
    best_action = None
    best_predicted_reward = float('-inf')
    
    for action in actions:
        predicted_next_state, predicted_reward = world_model.predict_outcome(
            agent_state.to_json(), json.dumps(action)
        )
        
        if predicted_reward > best_predicted_reward:
            best_predicted_reward = predicted_reward
            best_action = action
    
    # If world model has no good prediction, use the first action
    if best_action is None and actions:
        best_action = actions[0]
    
    # Step 7: Execute the selected action
    if best_action:
        print(f"Selected action: {best_action['type']} - {best_action['content']}")
        action_content = best_action['content']
        action_id = save_action(observation_id, best_action['type'], action_content)
        agent_state.update(new_action=best_action)
        
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
        
        # 8.6: Save reward to database
        context = json.dumps({
            "mood": mood_value,
            "explicit_feedback": user_feedback,
            "novelty": action_novelty
        })
        save_reward(action_id, composite_reward, "composite", context)
        
        # 8.7: Update agent state with reward
        agent_state.update(new_reward=composite_reward)
        
        # Step 9: Update world model
        world_model.update(
            agent_state.to_json(),
            json.dumps(best_action),
            agent_state.to_json(),  # Updated state after action
            composite_reward
        )
        
        print(f"Cycle completed. Composite reward: {composite_reward:.2f}")
    else:
        print("No action selected.")

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
    
    # Initialize agent state and world model
    agent_state = AgentState(user_id)
    world_model = SimpleWorldModel(user_id)
    
    # Display welcome message
    print(f"\nWelcome, {user_id}! I'll be observing and learning from our interactions.")
    print("Press Ctrl+C at any time to exit.")
    
    try:
        cycle_count = 0
        while True:
            # Perform a single interaction cycle
            await perform_interaction_cycle(user_id, agent_state, world_model)
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
                        world_model = SimpleWorldModel(user_id)
                except RuntimeError as e:
                    print(f"Error during user verification: {e}")
            
            # Wait 30 seconds between cycles (can be adjusted)
            print("\nWaiting for next cycle...")
            await asyncio.sleep(CYCLE_WAIT_TIME)
    
    except KeyboardInterrupt:
        print("\nExiting agentic insight loop.")
    except Exception as e:
        print(f"\nError in agentic insight loop: {e}")

# --- ENTRY POINT ---

if __name__ == "__main__":
    asyncio.run(agentic_insight_loop()) 