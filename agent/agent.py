from agent.neo4j_connector import Neo4jConnector
from agent.conversation_history import ConversationHistory
from agent.classifier import classify_intent
from agent.cypher_executor import execute_intent
from agent.synthesizer import synthesize_response


class KnowledgeGraphAgent:
    """
    Orchestrates the full pipeline:
    Input Capture → Conversation History → Parse & Classify
    → Cypher Execution → Synthesis Engine → Natural Language Response
    """

    def __init__(self):
        print("🔌 Connecting to Neo4j...")
        self.connector = Neo4jConnector()
        print("✅ Neo4j connected successfully.")
        self.history = ConversationHistory(max_turns=10)

    def run(self, user_input: str) -> str:
        """
        Process a single user message through the full agent pipeline.

        Args:
            user_input: The raw text from the user.

        Returns:
            A natural language response string.
        """
        # Step 1: Store user message
        self.history.add_message("user", user_input)
        history_text = self.history.format_for_prompt()

        # Step 2: Classify intent
        intent = classify_intent(user_input, history_text)

        # Step 3: Generate + execute Cypher
        result = execute_intent(self.connector, intent, user_input, history_text)

        # Step 4: Synthesize response
        response = synthesize_response(user_input, result)

        # Step 5: Store assistant response
        self.history.add_message("assistant", response)

        return response

    def close(self):
        """Clean up the Neo4j connection."""
        self.connector.close()
