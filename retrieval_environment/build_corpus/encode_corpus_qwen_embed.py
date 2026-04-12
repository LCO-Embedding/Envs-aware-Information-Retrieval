from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

data = load_dataset("./RAGBench-processed","corpus")["train"]
texts = data["text"]

model_path = "./embedding_checkpoints/Qwen3-Embedding-0.6B" 
device = "cuda:0"
BATCH_SIZE=64
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
model.eval()

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the last token's hidden state."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

all_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding Batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(normalized_embeddings.cpu())

embeddings_tensor = torch.cat(all_embeddings)
torch.save(embeddings_tensor,"corpus-embeddings-RAGBench-Qwen06B.pt")