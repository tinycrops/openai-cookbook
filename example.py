# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
from google import genai
from google.genai import types


def generate():
    client = genai.Client(api_key='GEMINI_API_KEY')

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""test weather Rochester, NY"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""The user is asking for the weather in \"Rochester, NY\". The available tool `getWeather` can provide this information. The tool requires a `city` argument. Therefore, I should call `getWeather` with `city` set to \"Rochester, NY\"."""),
                types.Part.from_function_call(
                    name="""getWeather""",
                    args={"city":"Rochester, NY"},
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name="""getWeather""",
                    response={
                      "output": """test""",
                    },
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""OK. The weather in Rochester, NY is test."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="getWeather",
                    description="gets the weather for a requested city",
                    parameters=genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        properties = {
                            "city": genai.types.Schema(
                                type = genai.types.Type.STRING,
                            ),
                        },
                    ),
                ),
            ])
    ]
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""test"""),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text if chunk.function_calls is None else chunk.function_calls[0])

if __name__ == "__main__":
    generate()
