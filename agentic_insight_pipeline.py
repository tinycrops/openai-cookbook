import os
import io
import base64
import asyncio
import datetime
import sqlite3
from typing import Optional, Dict, Any

import cv2
import PIL.Image
from google import genai
from google.genai import types
import numpy as np

# --- CONFIGURATION ---

DB_PATH = "agentic_insights.db"
MODEL = "models/gemini-2.0-flash"
EMBED_MODEL = "text-embedding-004"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- DATABASE SETUP ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            image BLOB,
            insights TEXT,
            agent_state TEXT,
            face_description TEXT,
            face_embedding TEXT
        )
    """)
    # Add face_embedding column if missing (migration)
    try:
        c.execute("ALTER TABLE observations ADD COLUMN face_embedding TEXT")
    except sqlite3.OperationalError:
        pass  # Already exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_embeddings (
            user_id TEXT PRIMARY KEY,
            avg_embedding TEXT,
            count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_observation(user_id: str, image_bytes: bytes, insights: str, agent_state: str, face_description: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO observations (user_id, timestamp, image, insights, agent_state, face_description) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, datetime.datetime.utcnow().isoformat(), image_bytes, insights, agent_state, face_description)
    )
    conn.commit()
    conn.close()

def get_last_insight(user_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT timestamp, insights, agent_state FROM observations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"timestamp": row[0], "insights": row[1], "agent_state": row[2]}
    return None

# --- IMAGE CAPTURE ---

def capture_image() -> bytes:
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

# --- GEMINI INSIGHT EXTRACTION (using generate_content) ---

async def extract_insights(image_bytes: bytes, last_insight: Optional[Dict[str, Any]]) -> str:
    system_instruction_text = (
        "You are an agentic assistant. "
        "Given a photo of a user, extract multi-level insights: "
        "1. Describe the user's appearance, mood, and environment. "
        "2. If previous insights are available, compare and note changes. "
        "3. Suggest actions or advice based on the user's state. "
        "Be concise, actionable, and context-aware."
    )

    # Combine instruction and prompt into user content for async API
    prompt_text = system_instruction_text + "\n\nAnalyze the user in this image."
    user_parts = [
        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)),
        types.Part(text=prompt_text)
    ]

    if last_insight:
         # Append context to the user parts if it exists
         user_parts.append(
              types.Part(text=
                  f"\n\nContext from previous observation ({last_insight['timestamp']}):\n{last_insight['insights']}"
              )
         )

    try:
        # Use the asynchronous client - arguments seem limited
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=user_parts # Pass combined parts directly
        )
        # Try accessing response.text directly first
        if hasattr(response, 'text') and response.text:
             return response.text
        # Fallback to checking candidates if .text is not available or empty
        elif response.candidates and response.candidates[0].content.parts:
             return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
             print("Warning: No text found in response.")
             return "" # Return empty string if no text found

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        # Consider more specific error handling or re-raising
        raise

# --- AGENTIC STATE MANAGEMENT (OPTIONAL) ---

def update_agent_state(last_state: Optional[str], new_insight: str) -> str:
    # For demo: just append new insight to state. In production, parse and update structured state.
    if last_state:
        return last_state + "\n---\n" + new_insight # Separator for clarity
    return new_insight

# --- FACE DESCRIPTION GENERATION ---

async def generate_face_description(image_bytes: bytes) -> str:
    prompt = (
        "Describe the face in this image for the purpose of recognizing the same person in the future, "
        "even if their appearance changes (e.g., beard/no beard, glasses/no glasses). "
        "Focus on stable features like face shape, eyes, etc."
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

# --- EMBEDDING UTILS ---
def get_face_embedding(face_description: str) -> np.ndarray:
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT avg_embedding, count FROM user_embeddings WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if row:
        avg_emb = np.fromstring(row[0], sep=',')
        count = row[1]
        updated_emb = (avg_emb * count + new_embedding) / (count + 1)
        c.execute("UPDATE user_embeddings SET avg_embedding = ?, count = ? WHERE user_id = ?",
                  (','.join(map(str, updated_emb.tolist())), count + 1, user_id))
    else:
        c.execute("INSERT INTO user_embeddings (user_id, avg_embedding, count) VALUES (?, ?, ?)",
                  (user_id, ','.join(map(str, new_embedding.tolist())), 1))
    conn.commit()
    conn.close()

def get_all_user_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, avg_embedding FROM user_embeddings")
    rows = c.fetchall()
    conn.close()
    user_embs = []
    for user_id, emb_str in rows:
        if emb_str:
            emb = np.fromstring(emb_str, sep=',')
            user_embs.append((user_id, emb))
    return user_embs

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- USER IDENTIFICATION ---

async def identify_user(image_bytes: bytes) -> str:
    new_desc = await generate_face_description(image_bytes)
    new_emb = get_face_embedding(new_desc)
    user_embs = get_all_user_embeddings()
    scored_candidates = []
    for user_id, avg_emb in user_embs:
        score = cosine_similarity(new_emb, avg_emb)
        scored_candidates.append((user_id, score))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top3 = scored_candidates[:3]
    print("Top 3 most likely users (embedding):")
    for idx, (user_id, score) in enumerate(top3, 1):
        print(f"  {idx}. user_id: {user_id} (cosine similarity: {score:.2f})")
    if top3 and top3[0][1] > 0.8:
        best_user = top3[0][0]
        print(f"Identified user '{best_user}' with confidence {top3[0][1]:.2f}.")
        return best_user
    else:
        print("No confident match found. Are you a new user? [y/N]")
        resp = input().strip().lower()
        if resp == 'y':
            print("Enter a new user_id:")
            user_id = input().strip()
            return user_id
        else:
            return "default_user"

# --- MAIN PIPELINE ---

async def agentic_insight_pipeline():
    init_db()
    print("Capturing image...")
    try:
        image_bytes = capture_image()
        print("Image captured.")
    except RuntimeError as e:
        print(f"Error capturing image: {e}")
        return

    print("Identifying user...")
    user_id = await identify_user(image_bytes)
    last = get_last_insight(user_id)
    print(f"Last insight found for user '{user_id}': {'Yes' if last else 'No'}")

    print("Generating face description...")
    face_description = await generate_face_description(image_bytes)
    face_embedding = get_face_embedding(face_description)

    print("Extracting insights...")
    insight = await extract_insights(image_bytes, last)

    if insight:
        print("Insights extracted.")
        agent_state = update_agent_state(last["agent_state"] if last else None, insight)
        save_observation(user_id, image_bytes, insight, agent_state, face_description)
        save_user_embedding(user_id, face_embedding)
        print("\n--- Agentic Insight ---")
        print(insight)
        print("-----------------------")
        print("Observation saved.")
    else:
        print("Failed to extract insights or no insight returned.")

# --- ENTRY POINT ---

if __name__ == "__main__":
    asyncio.run(agentic_insight_pipeline()) 