from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt, generation_config):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
