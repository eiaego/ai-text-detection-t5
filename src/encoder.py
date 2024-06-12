from transformers import T5EncoderModel, T5Tokenizer
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import numpy as np

class encoder():
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("Kijai/flan-t5-xl-encoder-only-bf16")
        self.llm_encoder = T5EncoderModel.from_pretrained("Kijai/flan-t5-xl-encoder-only-bf16", cache_dir='./')

    def encode(self, text, device):
        input_ids = self.tokenizer(

        text, padding='max_length', return_tensors="pt"

        ).input_ids
    
        input_ids = input_ids[:, :512]
        
        input_ids = input_ids.to(device)
        self.llm_encoder.to(device)
        outputs = self.llm_encoder(input_ids=input_ids)

        embeddings = outputs.last_hidden_state
        embeddings = embeddings.to('cpu')
        embeddings = embeddings.detach().numpy()[0]

        return embeddings