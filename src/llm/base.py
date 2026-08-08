from abc import ABC, abstractmethod


class ChatProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError