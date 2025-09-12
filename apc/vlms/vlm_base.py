from abc import ABC, abstractmethod
from torch import nn

class VLMBase(ABC, nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    @abstractmethod
    def load_model(self, config, device: str = "cuda"):
        """
        Load the model, tokenizer, and other required components based on the configuration.
        """
        pass

    @abstractmethod
    def generate_response(
        self,
        prompt,
        image=None,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
    ):
        """
        Run a single VQA
        """
        pass

    @abstractmethod
    def process_messages(
        self,
        messages,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
    ):
        """
        Process full conversation
        """
        pass