import torch
import torch.nn.functional as F
from tqdm import tqdm
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
from typing import List
from pydantic import BaseModel

app = FastAPI()

BATCH_SIZE = 128
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained('./embedding_checkpoints/minilm')
model = AutoModel.from_pretrained('./embedding_checkpoints/minilm').to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def get_embedding(request: EmbedRequest):
    if not request.texts:
        return {"embeddings": []}

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(request.texts), BATCH_SIZE), desc="Encoding Batches"):
            batch_texts = request.texts[i:i + BATCH_SIZE]
            
            batch_dict = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            
            embeddings = mean_pooling(outputs, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)        
            all_embeddings.append(embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings)

    return {"embeddings": embeddings_tensor.tolist()}