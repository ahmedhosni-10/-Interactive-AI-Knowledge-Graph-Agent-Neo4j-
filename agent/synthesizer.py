
import os
from llama_index.llms.groq import Groq


def synthesize_response(user_input: str, execution_result: dict) -> str:
    """
    Generate a natural language response based on the execution result.

    Args:
        user_input:        The original user message.
        execution_result:  Dict from cypher_executor containing:
                           - success (bool)
                           - intent  (str)
                           - cypher  (str)
                           - results (list) or error (str)

    Returns:
        A friendly, human-readable string response.
    """
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(model=model, api_key=api_key)

    intent = execution_result.get("intent", "unknown")
    success = execution_result.get("success", False)

    if not success:
        error = execution_result.get("error", "Unknown error")
        context = (
            f"The operation failed.\n"
            f"Error message: {error}\n"
            f"Attempted Cypher: {execution_result.get('cypher', 'N/A')}"
        )
    else:
        results = execution_result.get("results", [])
        context = (
            f"The operation completed successfully.\n"
            f"Operation type: {intent}\n"
            f"Cypher executed: {execution_result.get('cypher', 'N/A')}\n"
            f"Returned records: {results if results else '(No records returned — write operation)'}"
        )

    prompt = f"""You are a friendly and knowledgeable AI assistant for a Neo4j knowledge graph.
Generate a concise, natural, conversational response for the user.

User's original request: {user_input}
Operation context:
{context}

Guidelines:
- If it was a successful WRITE operation (add/edit/delete), confirm what was done clearly.
- If it was a successful READ operation (inquire), present the results in a readable format.
- If it failed, apologize briefly and mention the issue without showing raw error details.
- Keep the response concise and friendly.
- Do NOT include raw Cypher in your response.
"""

    response = llm.complete(prompt)
    return response.text.strip()
