from enum import Enum


class PlatformEnum(Enum):
    GROQ = "groq"
    OPENAI = "openai"


AI_MODELS = [
    {
        "name": "Gemma 2 9B",
        "description": "",
        "platform": PlatformEnum.GROQ,
        "model_name": "gemma2-9b-it",
        "developer": "Google",
        "details": {
            "request_per_min": 30,
            "request_per_day": 14400,
            "tokens_per_minute": 15000,
            "toekn_per_day": 500000,
        },
    },
    {
        "name": "Gemma 7B",
        "description": "",
        "platform": PlatformEnum.GROQ,
        "model_name": "gemma-7b-it",
        "developer": "Google",
        "details": {
            "request_per_min": 30,
            "request_per_day": 14400,
            "tokens_per_minute": 15000,
            "toekn_per_day": 500000,
        },
    },
]
