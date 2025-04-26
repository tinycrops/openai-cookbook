#!/usr/bin/env python3
"""
Example: Agentic LLM integration with agentic_info_systems for recording and fetching signals.
"""

import os

from google import genai
from google.genai import types

from signal_recorder import SignalRecorder


def run():
    # Retrieve Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Select model
    model = "gemini-2.5-flash-preview-04-17"

    # Initialize signal recorder
    recorder = SignalRecorder()

    # Define tools for function calling
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="recordSignal",
                    description="Record a signal with a name, numeric value, and optional metadata",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "name": genai.types.Schema(type=genai.types.Type.STRING),
                            "value": genai.types.Schema(type=genai.types.Type.NUMBER),
                            "metadata": genai.types.Schema(type=genai.types.Type.OBJECT),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="fetchSignals",
                    description="Fetch recent signals, optionally filtered by name and limit",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "name": genai.types.Schema(type=genai.types.Type.STRING),
                            "limit": genai.types.Schema(type=genai.types.Type.INTEGER),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="takePictureAndRecord",
                    description="Take a picture using the webcam and record it in the database with a name and optional metadata.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "name": genai.types.Schema(type=genai.types.Type.STRING),
                            "metadata": genai.types.Schema(type=genai.types.Type.OBJECT),
                        },
                        required=["name"]
                    ),
                ),
            ]
        )
    ]

    # Prepare system instruction
    system_parts = [
        types.Part.from_text(
            text=(
                "You are an agentic system that can record and fetch signals using the provided tools. "
                "When asked to record or fetch signals, respond with the appropriate function call."
            )
        )
    ]

    # Initial conversation: system instruction and user prompt
    contents = [
        types.Content(role="user", parts=system_parts)
    ]
    user_input = input("Enter your request (e.g., record or fetch signals): ")
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    )

    # Configure generation with tools
    config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )

    # Stream generation and handle function calls
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.function_calls:
            call = chunk.function_calls[0]
            name = call.name
            args = call.args or {}
            if name == "recordSignal":
                recorder.record_signal(
                    name=args.get("name"),
                    value=args.get("value"),
                    metadata=args.get("metadata"),
                )
                response = types.Content(
                    role="model",
                    parts=[types.Part.from_function_response(name=name, response={"status": "recorded"})]
                )
            elif name == "fetchSignals":
                signals = recorder.fetch_signals(
                    name=args.get("name"),
                    limit=args.get("limit", 10),
                )
                response = types.Content(
                    role="model",
                    parts=[types.Part.from_function_response(name=name, response={"signals": signals})]
                )
            elif name == "takePictureAndRecord":
                # Take a picture using the webcam and record it
                try:
                    import cv2
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        raise RuntimeError("Could not open webcam.")
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        raise RuntimeError("Failed to capture image from webcam.")
                    # Encode as JPEG
                    ret, buf = cv2.imencode('.jpg', frame)
                    if not ret:
                        raise RuntimeError("Failed to encode image.")
                    image_bytes = buf.tobytes()
                    recorder.record_image_signal(
                        name=args.get("name"),
                        image_bytes=image_bytes,
                        metadata=args.get("metadata"),
                    )
                    # Analyze the image with Gemini for captioning
                    from google.genai import types as gtypes
                    import base64
                    # Create a Gemini File object from bytes (inline image)
                    image_part = gtypes.Part.from_data(data=image_bytes, mime_type="image/jpeg")
                    caption_prompt = gtypes.Content(
                        role="user",
                        parts=[image_part, gtypes.Part.from_text("Describe this image.")]
                    )
                    caption_response = client.models.generate_content(
                        model=model,
                        contents=[caption_prompt],
                        config=types.GenerateContentConfig(response_mime_type="text/plain")
                    )
                    caption_text = caption_response.text.strip() if hasattr(caption_response, 'text') else "(No caption returned)"
                    response = types.Content(
                        role="model",
                        parts=[types.Part.from_function_response(name=name, response={"status": "picture recorded", "understanding": caption_text})]
                    )
                except Exception as e:
                    response = types.Content(
                        role="model",
                        parts=[types.Part.from_function_response(name=name, response={"error": str(e)})]
                    )
            else:
                response = types.Content(
                    role="model",
                    parts=[types.Part.from_function_response(name=name, response={"error": f"Unknown function {name}"})]
                )
            contents.append(response)
        else:
            print(chunk.text, end="")

    recorder.close()


if __name__ == "__main__":
    run()