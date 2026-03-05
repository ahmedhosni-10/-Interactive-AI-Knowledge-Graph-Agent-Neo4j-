import os
from llama_index.llms.groq import Groq


# ─────────────────────────────────────────────────────────────
# Intent-specific prompt templates
# ─────────────────────────────────────────────────────────────

_PROMPT_ADD = """\
You are an expert Neo4j Cypher generator. Generate a Cypher statement to CREATE or MERGE \
new nodes and/or relationships based on the user's request.

Rules:
- Use MERGE instead of CREATE to avoid duplicate nodes.
- Use meaningful, PascalCase labels (e.g., Person, City, Company).
- Use camelCase property names (e.g., name, age, foundedYear).
- Return ONLY the raw Cypher query — no explanation, no backticks, no markdown.

Conversation history:
{history}

User request: {user_input}
"""

_PROMPT_INQUIRE = """\
You are an expert Neo4j Cypher generator. Generate a Cypher MATCH ... RETURN statement \
to retrieve information that answers the user's question.

Rules:
- Always start with MATCH.
- Return meaningful properties or nodes.
- Return ONLY the raw Cypher query — no explanation, no backticks, no markdown.

Conversation history:
{history}

User question: {user_input}
"""

_PROMPT_EDIT = """\
You are an expert Neo4j Cypher generator. Generate a Cypher MATCH ... SET statement \
to update or correct existing information based on the user's request.

Rules:
- Use MATCH to find the existing node/relationship first.
- Use SET to update properties.
- Return ONLY the raw Cypher query — no explanation, no backticks, no markdown.

Conversation history:
{history}

User request: {user_input}
"""

_PROMPT_DELETE = """\
You are an expert Neo4j Cypher generator. Generate a Cypher MATCH ... DETACH DELETE statement \
to remove nodes and/or relationships based on the user's request.

Rules:
- Use MATCH to find the target node/relationship first.
- Use DETACH DELETE for nodes (removes all relationships automatically).
- Return ONLY the raw Cypher query — no explanation, no backticks, no markdown.

Conversation history:
{history}

User request: {user_input}
"""

_PROMPTS = {
    "add": _PROMPT_ADD,
    "inquire": _PROMPT_INQUIRE,
    "edit": _PROMPT_EDIT,
    "delete": _PROMPT_DELETE,
}


# ─────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────

def _generate_cypher(intent: str, user_input: str, history: str) -> str:
    """Use the LLM to generate a Cypher query for the given intent."""
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(model=model, api_key=api_key)

    template = _PROMPTS[intent]
    prompt = template.format(
        history=history if history else "(No prior conversation)",
        user_input=user_input,
    )

    response = llm.complete(prompt)
    cypher = response.text.strip()

    # Strip any accidental markdown code fences
    for fence in ["```cypher", "```Cypher", "```CYPHER", "```"]:
        cypher = cypher.replace(fence, "")
    cypher = cypher.strip()

    return cypher


def execute_intent(connector, intent: str, user_input: str, history: str) -> dict:
    """
    Generate and execute the appropriate Cypher query for a given intent.

    Args:
        connector:   Neo4jConnector instance.
        intent:      One of "add", "inquire", "edit", "delete".
        user_input:  The user's original message.
        history:     Formatted conversation history string.

    Returns:
        {
            "success": bool,
            "intent":  str,
            "cypher":  str,
            "results": list  (on success),
            "error":   str   (on failure),
        }
    """
    cypher = _generate_cypher(intent, user_input, history)

    try:
        results = connector.run_query(cypher)
        return {
            "success": True,
            "intent": intent,
            "cypher": cypher,
            "results": results,
        }
    except Exception as e:
        return {
            "success": False,
            "intent": intent,
            "cypher": cypher,
            "error": str(e),
        }
