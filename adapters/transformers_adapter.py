import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from adapters.base import ModelAdapter


class TransformersAdapter(ModelAdapter):
    def __init__(self, base_model, adapter_path=None,
                 device_map="auto", trust_remote_code=False):
        self.base_model_path = base_model
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        if self.adapter_path:
            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path
            )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt, generation_config):
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, ValueError):
            rendered = prompt

        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        gen = dict(generation_config or {})
        gen.setdefault("max_new_tokens", 256)
        gen.setdefault("do_sample", False)
        if not gen.get("do_sample", False):
            gen.pop("temperature", None)
            gen.pop("top_p", None)
        if self.tokenizer.pad_token_id is not None:
            gen.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif self.tokenizer.eos_token_id is not None:
            gen.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        output = self.model.generate(**encoded, **gen)
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

    def close(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
