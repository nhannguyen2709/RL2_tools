from .base import tokenize_messages
from .rm import RMDataset


class DPODataset(RMDataset):
    
    def tokenize_messages_completion(self, messages, completion):

        ex = tokenize_messages(
            self.tokenizer,
            messages + [{"role": "assistant", "content": completion}]
        )
        return {k: v[:self.max_length] for k, v in ex.items()}