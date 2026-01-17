import os
from typing import Optional

class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            from together import Together
            self.client = Together(api_key=api_key)
        else:
            raise ValueError("provider must be one of: openai, together")

    def chat(self, system: str, user: str) -> str:
        base_params = {
            "model": self.model,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        resp = self.client.chat.completions.create(**base_params)
        return (resp.choices[0].message.content or "").strip()
