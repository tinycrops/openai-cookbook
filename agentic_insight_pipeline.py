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

# --- CONFIGURATION ---

DB_PATH = "agentic_insights.db"
MODEL = "models/gemini-2.0-flash"
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
            agent_state TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_observation(user_id: str, image_bytes: bytes, insights: str, agent_state: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO observations (user_id, timestamp, image, insights, agent_state) VALUES (?, ?, ?, ?, ?)",
        (user_id, datetime.datetime.utcnow().isoformat(), image_bytes, insights, agent_state)
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

# --- MAIN PIPELINE ---

async def agentic_insight_pipeline(user_id: str):
    init_db()
    print("Capturing image...")
    try:
        image_bytes = capture_image()
        print("Image captured.")
    except RuntimeError as e:
        print(f"Error capturing image: {e}")
        return

    last = get_last_insight(user_id)
    print(f"Last insight found for user '{user_id}': {'Yes' if last else 'No'}")

    print("Extracting insights...")
    insight = await extract_insights(image_bytes, last)

    if insight:
        print("Insights extracted.")
        agent_state = update_agent_state(last["agent_state"] if last else None, insight)
        save_observation(user_id, image_bytes, insight, agent_state)
        print("\n--- Agentic Insight ---")
        print(insight)
        print("-----------------------")
        print("Observation saved.")
    else:
        print("Failed to extract insights or no insight returned.")

# --- ENTRY POINT ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, required=True, help="User ID for the agentic stream")
    args = parser.parse_args()
    asyncio.run(agentic_insight_pipeline(args.user)) 