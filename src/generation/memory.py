from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    role: str    # "user" or "assistant"
    content: str


class ConversationMemory:
    """Manages multi-turn conversation history."""

    def __init__(self, max_turns: int = 5):
        self.history: List[ConversationTurn] = []
        self.max_turns = max_turns  # Limit history to avoid token overflow

    def add_user_message(self, message: str):
        self.history.append(ConversationTurn(role="user", content=message))
        self._trim()

    def add_assistant_message(self, message: str):
        self.history.append(ConversationTurn(role="assistant", content=message))
        self._trim()

    def _trim(self):
        """Keep only the last N turns to stay within LLM context limits."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def format_history(self) -> str:
        """Format history as a readable string for the prompt."""
        if not self.history:
            return "No previous conversation."
        lines = []
        for turn in self.history:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def clear(self):
        """Reset conversation history."""
        self.history = []

    def __len__(self):
        return len(self.history)