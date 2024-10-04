from llm.base import BaseLLM
from llm.groq import GroqAILLM
from llm.huggingface import HuggingFaceAILLM

def LLM_Factory(platform: str) -> BaseLLM:
    if platform == "GROQ":
        return GroqAILLM()
    elif platform == "HUGGINGFACE":
        return HuggingFaceAILLM()
    raise Exception("Unsupported platform specified.")