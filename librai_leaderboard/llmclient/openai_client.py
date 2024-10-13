import time
from openai import OpenAI
from .base import BaseClient


support_models = {
    "gpt-4o":"gpt-4o",
    "gpt-4o-mini":"gpt-4o-mini",
    "gpt-3.5-turbo":"gpt-3.5-turbo",
}

class OpenAI_Client(BaseClient):
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_config: dict = None,
        max_requests_per_minute=200,
        request_window=60,
    ):
        super().__init__(model, api_config, max_requests_per_minute, request_window)
        self.client_name = "OpenAI"
        assert self.api_config["OPENAI_API_KEY"] != ""
        self.client = OpenAI(api_key=self.api_config["OPENAI_API_KEY"])
        self.support_models = support_models

    def _call(self, messages: str, **kwargs):
        seed = kwargs.get("seed", 42)  # default seed is 42
        assert type(seed) is int, "Seed must be an integer."

        num_retries = kwargs.get("num_retries", 1)
        post_check_function = kwargs.get("post_check_function", None)
        response_format = kwargs.get("response_format", None)  # JSON mode: response_format={ "type": "json_object" },

        r = ""
        for _ in range(num_retries):
            try:
                response = self.client.chat.completions.create(
                    response_format=response_format,
                    seed=seed,
                    model=self.support_models[self.model],
                    messages=messages,
                )
                r = response.choices[0].message.content

                if post_check_function is not None and post_check_function(r):
                    break
            except Exception as e:
                print("Error: ", e)
                time.sleep(1)

        if r == "":
            print(f"{self.client_name} Client Warning: Empty response from LLM Client call. To debug, you may go to llmclient/openai_client.py line 34, and remove the try-except block.")

        # if response in locals() and hasattr(response, "usage"):
        #     self._log_usage(usage_dict=response.usage)
        # else:
        #     print("Warning: ChatGPT API Usage is not logged.")

        return r

    def _log_usage(self, usage_dict):
        try:
            self.usage.prompt_tokens += usage_dict.prompt_tokens
            self.usage.completion_tokens += usage_dict.completion_tokens
        except:  # noqa E722
            print("Warning: prompt_tokens or completion_token not found in usage_dict")

    def get_request_length(self, messages):
        # TODO: check if we should return the len(menages) instead
        return 1

    def construct_message_list(
        self,
        prompt_list: list[str],
        system_role: str = "You are a helpful assistant designed to output JSON.",
    ):
        messages_list = list()
        for prompt in prompt_list:
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ]
            messages_list.append(messages)
        return messages_list
