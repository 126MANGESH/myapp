#!/usr/bin/env python3
"""
Flask Chat App — Fully Compatible With Your index.html
RAG Disabled • Pure Chat UI Backend
Now Integrated with Azure OpenAI for Real AI Responses
"""

import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from openai import OpenAI  # Requires: pip install openai

app = Flask(__name__)

# In-memory chat history (simple; for production, use Redis or DB for multi-user)
chat_history = []

# Initialize OpenAI client (uses environment variables for Azure OpenAI)
# Set these env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME
client = None
try:
    client = OpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-15-preview"  # Stable version for Azure OpenAI
    )
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}. Falling back to mock mode.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Your frontend sends:
       { "message": "hello" }

    Backend must return:
       { "response": "hello reply..." }
    """
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message'"}), 400

    user_msg = data["message"].strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    chat_history.append({"role": "user", "content": user_msg})

    # Limit history to last 20 exchanges to avoid token limits (10 turns)
    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    # Generate reply
    if client:
        try:
            # Prepare messages: full history + system prompt
            messages = [
                {"role": "system", "content": "You are a helpful, concise assistant. Respond naturally and keep replies under 200 words."}
            ] + chat_history

            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")  # Updated default to your deployment
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=200,  # Limit response length
                temperature=0.7  # Balanced creativity
            )
            reply = response.choices[0].message.content.strip()

            if not reply:
                reply = "Hmm, that didn't generate a response. Try rephrasing!"

        except Exception as e:
            # Graceful fallback on API errors (e.g., invalid key, rate limits)
            print(f"OpenAI Error: {e}")
            reply = f"Sorry, AI service is temporarily unavailable: {str(e)[:100]}. Using mock mode."
            # Mock as fallback
            reply = f"🤖 AI: (Fallback) You said → {user_msg}"
    else:
        # Mock mode if client not initialized
        reply = f"🤖 AI: (Mock Mode - Set env vars for real AI) You said → {user_msg}"

    chat_history.append({"role": "assistant", "content": reply})

    return jsonify({"response": reply}), 200

@app.route("/reset", methods=["POST"])
def reset():
    """Clear chat history"""
    chat_history.clear()
    return jsonify({"status": "cleared"}), 200

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "messages": len(chat_history),
        "rag_enabled": False,
        "ai_mode": "real" if client else "mock",
        "time": datetime.utcnow().isoformat() + "Z"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    print(f"🚀 Running on http://0.0.0.0:{port} (AI Mode: {'Real' if client else 'Mock'})")
    app.run(host="0.0.0.0", port=port, debug=debug)