"""
conversation_history.py
Manages in-memory conversation history for multi-turn context.
"""


class ConversationHistory:
    """Stores and formats the conversation turns for LLM prompts."""

    def __init__(self, max_turns: int = 10):
        self.history: list[dict] = []
        self.max_turns = max_turns  # Keep the last N messages for context

    def add_message(self, role: str, content: str):
        """Append a new message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        """Return the full conversation history list."""
        return self.history

    def format_for_prompt(self) -> str:
        """
        Format recent conversation history as a readable string for LLM prompts.
        Only includes the last `max_turns` messages to keep prompts concise.
        """
        recent = self.history[-self.max_turns:]
        lines = []
        for msg in recent:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Returns True if there is no conversation history yet."""
        return len(self.history) == 0
