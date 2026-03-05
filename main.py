import os
import sys
from dotenv import load_dotenv

# ── Fix Unicode/emoji output on Windows terminals (cp1252 → utf-8) ──────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)  # Always use .env values, even if shell has stale env vars

def _check_env():
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.getenv("NEO4J_URI"):
        missing.append("NEO4J_URI")
    if not os.getenv("NEO4J_PASSWORD"):
        missing.append("NEO4J_PASSWORD")

    if missing:
        print("❌ Missing required environment variables in .env:")
        for var in missing:
            print(f"   • {var}")
        print("\nPlease update your .env file and try again.")
        sys.exit(1)


def _print_banner():
    print("🧠 AI Knowledge Graph Agent")
    print()


def main():
    _check_env()
    _print_banner()
    from agent.agent import KnowledgeGraphAgent

    agent = KnowledgeGraphAgent()
    print()

    try:
        while True:
            # ── Read user input ──────────────────────────────────────
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nAgent: Goodbye! 👋")
                break

            if not user_input:
                continue

            # ── Built-in commands ────────────────────────────────────
            if user_input.lower() in ("exit", "quit", "bye"):
                print("Agent: Goodbye! Happy graphing! 👋")
                break

            if user_input.lower() == "history":
                history = agent.history.get_history()
                if not history:
                    print("(No conversation history yet.)\n")
                else:
                    print("\n── Conversation History ──────────────────────────────")
                    for msg in history:
                        prefix = "You  " if msg["role"] == "user" else "Agent"
                        print(f"  [{prefix}] {msg['content']}")
                    print("─────────────────────────────────────────────────────\n")
                continue

            if user_input.lower() == "clear":
                agent.history.history.clear()
                print("Agent: Conversation history cleared. ✅\n")
                continue

            # ── Run the agent pipeline ───────────────────────────────
            print("Agent: ⏳ Processing...")
            try:
                response = agent.run(user_input)
                print(f"\rAgent: {response}\n")
            except Exception as e:
                print(f"\rAgent: ⚠️  An error occurred: {e}\n")

    finally:
        agent.close()
        print("Neo4j connection closed.")


if __name__ == "__main__":
    main()
