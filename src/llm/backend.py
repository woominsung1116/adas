from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return raw response string."""
        ...

    def generate_raw(self, prompt: str) -> str:
        """Send prompt without schema enforcement. Defaults to generate()."""
        return self.generate(prompt)
