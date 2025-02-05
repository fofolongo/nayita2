# app.py
import os
import subprocess
import shutil
import requests
from flask import Flask, request, jsonify, send_from_directory
import openai

app = Flask(__name__, static_folder='.', static_url_path='')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure ffmpeg is installed and accessible.
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path is None or not os.path.exists(ffmpeg_path):
        raise EnvironmentError("ffmpeg not found in PATH. Please install ffmpeg and add it to your system PATH, or set the FFMPEG_PATH environment variable to the full path of the ffmpeg executable.")

# Global conversation history with an initial system prompt in Spanish.
conversation = [
    {"role": "system", "content": (
        """
        you are the best smart ai assintant in the universe, you can remember things, calculate
        always answer in spanish
        you can remember things, to dos, tasks
        you can search the internet
        always be concrete
        dont give me instructions unless i told you to
        you will have 2 lists to remember, tareas and gastos, you are able to categorize them
        if this is understood always refer to me as fofo and greet me as hi fofo
        """
    )}
]

def internet_search(query):
    # Replace this with actual search API integration if available.
    # For demonstration, this function returns a simulated search result.
    search_api_key = os.getenv("SEARCH_API_KEY")
    search_endpoint = os.getenv("SEARCH_API_ENDPOINT")  # e.g., Bing Search API endpoint
    if search_api_key and search_endpoint:
        params = {"q": query, "count": 3}
        headers = {"Ocp-Apim-Subscription-Key": search_api_key}
        response = requests.get(search_endpoint, params=params, headers=headers)
        if response.status_code == 200:
            results = response.json()
            snippets = []
            for item in results.get("webPages", {}).get("value", []):
                snippets.append(item.get("snippet", ""))
            return "\n".join(snippets)
        else:
            return "No se pudieron obtener resultados de búsqueda."
    else:
        return f"Resultados simulados para la búsqueda: {query}"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No se proporcionó archivo de audio"}), 400
    audio_file = request.files['audio']
    input_filename = "temp_input.webm"
    output_filename = "temp_output.wav"
    audio_file.save(input_filename)
    try:
        result = subprocess.run(
            [ffmpeg_path, "-y", "-i", input_filename, output_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            os.remove(input_filename)
            return jsonify({"error": "La conversión falló: " + result.stderr.decode("utf-8")}), 500
        with open(output_filename, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        user_text = transcript["text"]
        # Add the user's transcribed text to conversation history.
        conversation.append({"role": "user", "content": user_text})
        # Perform an internet search using the transcribed text to add context.
        search_results = internet_search(user_text)
        conversation.append({"role": "system", "content": f"Resultados de búsqueda en internet:\n{search_results}"})
        # Get chat completion using the updated conversation history.
        chat_response = openai.ChatCompletion.create(
            model="o3-mini",
            messages=conversation
        )
        assistant_text = chat_response["choices"][0]["message"]["content"]
        conversation.append({"role": "assistant", "content": assistant_text})
        return jsonify({"transcript": user_text, "assistant": assistant_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
