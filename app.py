# app.py
import os
import subprocess
import shutil
from flask import Flask, request, jsonify, send_from_directory
import openai

app = Flask(__name__, static_folder='.', static_url_path='')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure ffmpeg is installed and accessible.
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path is None or not os.path.exists(ffmpeg_path):
        raise EnvironmentError("ffmpeg not found in PATH. Please install ffmpeg and add it to your system PATH, "
                               "or set the FFMPEG_PATH environment variable to the full path of the ffmpeg executable.")

# Global conversation history with an initial system prompt.
conversation = [
    {"role": "system", "content": "You are ChatGPT. You are helpful."}
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
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
            return jsonify({"error": "Conversion failed: " + result.stderr.decode("utf-8")}), 500
        with open(output_filename, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        user_text = transcript["text"]
        # Add the transcribed text as a user message in the conversation
        conversation.append({"role": "user", "content": user_text})
        # Get ChatGPT response using the conversation history
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        assistant_text = chat_response["choices"][0]["message"]["content"]
        # Append assistant response to conversation history
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
    # Change host to '0.0.0.0' so that the server is accessible from other devices in your network.
    app.run(host='0.0.0.0', debug=True)
