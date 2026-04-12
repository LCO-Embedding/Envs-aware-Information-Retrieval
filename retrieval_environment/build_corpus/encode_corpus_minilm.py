from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

data = load_dataset("./RAGBench-filtered","corpus")["train"]
texts = data["text"]

BATCH_SIZE = 256
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained('./embedding_checkpoints/minilm')
model = AutoModel.from_pretrained('./embedding_checkpoints/minilm').to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


all_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding Batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        batch_dict = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        
        embeddings = mean_pooling(outputs, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)        
        all_embeddings.append(embeddings.cpu())

embeddings_tensor = torch.cat(all_embeddings)
torch.save(embeddings_tensor,"corpus-embeddings-RAGBench-minilm.pt")