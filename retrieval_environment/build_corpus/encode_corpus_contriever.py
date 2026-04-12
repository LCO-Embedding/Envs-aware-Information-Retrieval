
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

data = load_dataset("./RAGBench-filtered","corpus")["train"]
texts = data["text"]

BATCH_SIZE = 256
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained('./embedding_checkpoints/contriever')
model = AutoModel.from_pretrained('./embedding_checkpoints/contriever').to(device)

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

all_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding Batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        batch_dict = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        
        embeddings = mean_pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
        all_embeddings.append(embeddings.cpu())

embeddings_tensor = torch.cat(all_embeddings)
torch.save(embeddings_tensor,"corpus-embeddings-RAGBench-contriever.pt")