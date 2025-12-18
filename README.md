# Pawty 🐾

A multi-module pet interaction platform featuring a frontend web application, AI image style transformation service, RAG chat advisor, and supporting data assets.

## Repository Structure

```
Pawty/
├── frontend/
│   └── pawty-web/               # Frontend single-page application (index.html, static assets)
├── backend/
│   ├── image-style/             # Stable Diffusion image style transformation FastAPI service
│   └── rag-service/             # RAG + LLM advisor service (FastAPI + LangChain)
├── data/
│   ├── knowledge/               # JSONL knowledge base data
│   └── vectorstores/            # Persistent Chroma indexes (auto-generated/updated after running)
└── README.md
```

### Module Description

- **frontend/pawty-web**  
  Originally the `project-Pawty` folder. Simply open `index.html` for local preview; for deployment, use any static hosting service (Vercel, Netlify, Nginx, etc.).

- **backend/image-style**  
  AI transformation service based on Stable Diffusion Img2Img, providing `POST /stylize` endpoint to handle image uploads and style generation for the frontend "transformation" feature.

- **backend/rag-service**  
  RAG (Retrieval-Augmented Generation) advisor for the "chat" module. Uses Chroma as the vector store and OpenAI as the LLM, exposing `POST /ask` endpoint.

- **data**  
  - `knowledge/`: Stores original JSONL knowledge documents (prefix can be freely extended).  
  - `vectorstores/`: Stores built vector databases, e.g., `chroma_pawty/`. `rag_build.py --build` will write/overwrite here.

## Quick Start

### 1. Frontend

```bash
# Simply double-click or open with any static server
open frontend/pawty-web/index.html
```

When deploying to production, remember to update the `API_ENDPOINTS` in `frontend/pawty-web/index.html` with the backend domain.

### 2. Image Style Service (Transformation)

```bash
cd backend/image-style
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8011   # Optional: --lazy (load model on first request)
```

Optional environment variables:
- `PET_IMAGE_MODEL_ID`: Custom Stable Diffusion model (default: `runwayml/stable-diffusion-v1-5`)
- `HUGGINGFACE_TOKEN`: Required if the model needs HuggingFace authentication

### 3. RAG Chat Advisor (Chat)

```bash
cd backend/rag-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
OPENAI_API_KEY=sk-xxx python rag_build.py --build          # Generate vector store
OPENAI_API_KEY=sk-xxx python rag_build.py --serve --port 8001
```

After the service starts, the frontend will call `http://localhost:8001/ask` to get pet health advice.

### 4. Data and Vector Store

- Place new JSONL documents in `data/knowledge/`, then run `rag_build.py --build` again to update.
- Generated Chroma data is located in `data/vectorstores/chroma_pawty/`, which can be backed up or cleaned as needed.

## Common Workflow

1. Start the RAG service and image style service.
2. Open the frontend page for integration testing.
3. If knowledge data is updated, run `--build` again.

## License

MIT License - Contributions and improvements welcome. 🧡
