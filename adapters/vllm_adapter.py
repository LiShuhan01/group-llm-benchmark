import requests
from adapters.base import ModelAdapter


class VLLMAdapter(ModelAdapter):
    def __init__(self, api_base, served_model_name, api_key="EMPTY",
                 timeout=300):
        self.api_base = api_base.rstrip("/")
        self.served_model_name = served_model_name
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.session = requests.Session()

    def load(self):
        response = self.session.get(
            self.api_base + "/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=15,
        )
        response.raise_for_status()

    def generate(self, prompt, generation_config):
        cfg = dict(generation_config or {})
        payload = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.get("max_new_tokens", 256),
            "temperature": cfg.get("temperature", 0.0)
                if cfg.get("do_sample", False) else 0.0,
        }
        if cfg.get("do_sample", False) and "top_p" in cfg:
            payload["top_p"] = cfg["top_p"]

        response = self.session.post(
            self.api_base + "/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()

    def close(self):
        self.session.close()
