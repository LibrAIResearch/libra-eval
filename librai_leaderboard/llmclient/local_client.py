
import openai
from .openai_client import OpenAI_Client

support_models = {"llama-3-8b-instruct":"meta-llama/Meta-Llama-3-8B-Instruct",
                  "llama-3-70b-instruct":"meta-llama/Meta-Llama-3-70B-Instruct",
                  "phi-3-small-8k-instruct":"microsoft/Phi-3-small-8k-instruct",
                  "falcon-7b-instruct":"tiiuae/falcon-7b-instruct",
                  "qwen1.5-7b-chat":"Qwen/Qwen1.5-7B-Chat",
                  "llama-2-13b":"meta-llama/Llama-2-13b-chat-hf",
                  }


class Local_Client(OpenAI_Client):
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_config: dict = None,
        max_requests_per_minute=20,
        request_window=10,
    ):
        super().__init__(model, api_config, max_requests_per_minute, request_window)
        self.client_name = "Local"

        self.client = openai.OpenAI(
            base_url="http://0.0.0.0:8000/v1",
            api_key="token-abc123",
        )
        self.support_models = support_models
