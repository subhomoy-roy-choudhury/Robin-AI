from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        
    @abstractmethod
    def load_model(self, model_name: str) -> str:
        pass

    @abstractmethod
    def format_messages(self, messages: list) -> str:
        pass