from enum import Enum

class UserRole(Enum):
    MEMBER = "member"
    GUEST = "guest"

class PlatformEnum(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

AI_MODELS = [
    {
        "name": "Phi-3.5-mini",
        "description": "",
        "platform": PlatformEnum.HUGGINGFACE,
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "developer": "Microsoft",
        "details": {},
        "openllm_model_name": "microsoft/Phi-3.5-mini-instruct",
        "references": ["https://huggingface.co/microsoft/Phi-3.5-mini-instruct"],
    },
    {
        "name": "Gemma 2 9B",
        "description": "",
        "platform": PlatformEnum.GROQ,
        "model_name": "gemma2-9b-it",
        "developer": "Google",
        "openllm_model_name": "ehristoforu/Gemma2-9b-it-train6",
        "details": {
            "request_per_min": 30,
            "request_per_day": 14400,
            "tokens_per_minute": 15000,
            "token_per_day": 500000,
        },
    },
    {
        "name": "Gemma 7B",
        "description": "",
        "platform": PlatformEnum.GROQ,
        "model_name": "gemma-7b-it",
        "developer": "Google",
        "openllm_model_name": "google/gemma-7b-it",
        "details": {
            "request_per_min": 30,
            "request_per_day": 14400,
            "tokens_per_minute": 15000,
            "token_per_day": 500000,
        },
    },
]