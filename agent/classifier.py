import os
import json

from llama_index.llms.groq import Groq


def classify_intent(user_input: str, history: str) -> str:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(model=model, api_key=api_key)

    prompt = f"""You are an intent classifier for a knowledge graph chatbot.
Your job is to read the user's latest message (with conversation context) and classify
it into EXACTLY ONE of the following intents:

  - "add"     : The user wants to store new information, add nodes, or create relationships.
  - "inquire" : The user wants to search for, find, or ask about information.
  - "edit"    : The user wants to update, change, or correct existing information.
  - "delete"  : The user wants to remove or delete existing nodes or relationships.

--- Conversation History ---
{history if history else "(No prior conversation)"}

--- Current User Message ---
{user_input}

Respond with ONLY a valid JSON object and nothing else, like this:
{{"intent": "<add|inquire|edit|delete>"}}
"""

    response = llm.complete(prompt)
    raw = response.text.strip()

    # Parse the JSON response
    try:
        data = json.loads(raw)
        intent = data.get("intent", "inquire").lower().strip()
    except (json.JSONDecodeError, AttributeError):
        # Fallback: scan the raw text for keywords
        raw_lower = raw.lower()
        if "add" in raw_lower or "create" in raw_lower or "store" in raw_lower:
            intent = "add"
        elif "edit" in raw_lower or "update" in raw_lower or "change" in raw_lower:
            intent = "edit"
        elif "delete" in raw_lower or "remove" in raw_lower:
            intent = "delete"
        else:
            intent = "inquire"

    # Validate and sanitize
    valid_intents = {"add", "inquire", "edit", "delete"}
    return intent if intent in valid_intents else "inquire"
