#!/usr/bin/env python3
"""
Flask Chat App — Fully Compatible With Your index.html
RAG Disabled • Pure Chat UI Backend
"""

import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# In-memory chat history
chat_history = []

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

    user_msg = data["message"]
    chat_history.append(("user", user_msg))

    # ------------- MOCK AI REPLY -------------
    # Replace later with real OpenAI/Azure code
    reply = f"🤖 AI: You said → {user_msg}"

    chat_history.append(("assistant", reply))

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
        "time": datetime.utcnow().isoformat() + "Z"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
