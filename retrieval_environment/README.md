## Retrieval Environments

Retrieval environments are mostly defined by `corpus quality/quantify` and `retriever behaviors`. This part of the repo builds these two components.


### Serve Retrievers

A few retrievers are defined under `serve_retrievers`, and are faciliated by `fastapi` nad `uvicorn`.

```
source activate envs/serve
cd serve_retrievers
```

To serve Contriever for example, run:

```
uvicorn serve_contriever:app --host 0.0.0.0 --port 8000
```

all-MiniLM-L6-v2 & Qwen3-Embedding similarly:
```
uvicorn serve_minilm:app --host 0.0.0.0 --port 8000
```

```
uvicorn serve_qwen_embed:app --host 0.0.0.0 --port 8000
```

You can easily extend any of these examplar serving code to your custom retrievers using their encoding logic.