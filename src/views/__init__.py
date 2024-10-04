from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> str:
        pass

    @abstractmethod
    def format_messages(self, messages: list) -> str:
        pass