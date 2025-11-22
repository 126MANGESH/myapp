import os
import sys
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from openai import AzureOpenAI

# 1. Load Environment Variables
load_dotenv()

app = Flask(__name__)
# Secure secret key for session management
app.secret_key = os.urandom(24)

# 2. Configuration & Safety Checks
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

print("\n=== SAGEALPHA SYSTEM STARTUP ===")
print(f"Target Endpoint: {ENDPOINT}")
print(f"Deployment:      {DEPLOYMENT}")
print(f"API Version:     {API_VERSION}")

if not API_KEY or "PASTE_YOUR" in API_KEY:
    print("❌ ERROR: API Key is missing in .env file.")
    sys.exit(1)

# 3. Initialize Azure Client
try:
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION
    )
    print("✅ Azure Client Initialized Successfully")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    sys.exit(1)

# --- Routes ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = (request.json.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Initialize chat history in session if not exists
    if 'history' not in session:
        session['history'] = [
            {"role": "system", "content": "You are SageAlpha, an advanced financial AI assistant."}
        ]
    
    # Add User Message to History
    history = session['history']
    history.append({"role": "user", "content": user_msg})

    try:
        # Call Azure OpenAI (Chat Completion)
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=history,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95
        )

        # Extract AI Message
        ai_msg = response.choices[0].message.content
        
        # Add AI Response to History
        history.append({"role": "assistant", "content": ai_msg})
        session['history'] = history # Save updated history

        return jsonify({"response": ai_msg})

    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    session.pop('history', None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)