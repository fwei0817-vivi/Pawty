"""
rag_build.py
Pawty RAG Backend:
- Build Chroma vector DB from JSONL files
- Provide /ask API using FastAPI (RAG + OpenAI)
- Report metrics to rag_monitor.py (/log)

Usage:
  # 1) 构建向量库（只需在数据更新后执行一次）
  python rag_build.py --build

  # 2) 启动服务（提供 /ask 接口）
  python rag_build.py --serve --port 8001
"""

import os
import glob
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI


# ============= 全局配置 =============

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "knowledge"
VECTORSTORE_DIR = REPO_ROOT / "data" / "vectorstores" / "chroma_pawty"

DATA_GLOB = str(DATA_DIR / "*.jsonl")          # 你的知识数据（爬虫输出）
PERSIST_DIR = str(VECTORSTORE_DIR)      # 向量库持久化目录
COLLECTION_NAME = "pawty_v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5

# FastAPI app
app = FastAPI(title="Pawty RAG API", version="1.0.0")
# 开放跨域，便于本地前端 (8080) 调用 8001
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局向量库实例
vectordb: Optional[Chroma] = None

# 加载 .env（可选）
load_dotenv()

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not set. /ask 会报错，请先在环境中设置。")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============= 数据加载 & 建库 =============

def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    records = []
    for fp in glob.glob(pattern):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def record_to_doc(rec: Dict[str, Any]) -> Optional[Document]:
    text = rec.get("summary") or rec.get("content") or ""
    text = text.strip()
    if not text:
        return None

    meta_keys = [
        "title", "topic", "species", "breed", "life_stage",
        "source_name", "source_url", "source_date", "crawl_date",
    ]
    metadata = {k: rec.get(k) for k in meta_keys if rec.get(k) is not None}
    return Document(page_content=text, metadata=metadata)


def build_index():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("🔍 Loading JSONL data from", DATA_GLOB)
    raw = load_jsonl_files(DATA_GLOB)
    docs: List[Document] = []
    for r in raw:
        d = record_to_doc(r)
        if d:
            docs.append(d)

    if not docs:
        raise RuntimeError("No documents found under ./data. 请先准备 JSONL 数据。")

    print(f"📚 Loaded {len(docs)} docs. Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks: List[Document] = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            chunks.append(Document(page_content=c, metadata=d.metadata))

    print(f"🔢 Total chunks: {len(chunks)}. Building embeddings with {EMBED_MODEL} ...")
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("🧱 Creating Chroma DB ...")
    # 先清空旧库
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=PERSIST_DIR,
    )
    try:
        db.delete_collection()
    except Exception:
        pass

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=PERSIST_DIR,
    )
    db.add_documents(chunks)
    db.persist()
    print(f"✅ Vector DB built & persisted at {PERSIST_DIR}")


def load_vectordb() -> Chroma:
    """在 serve 模式中加载向量库到全局变量"""
    global vectordb
    if vectordb is not None:
        return vectordb

    print("🔍 Loading Chroma vector DB from disk ...")
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=PERSIST_DIR,
    )
    print("✅ Vector DB loaded.")
    return vectordb


# ============= RAG 检索 + 生成 =============

def filter_by_metadata(
    docs: List[Document],
    species: Optional[str] = None,
    breed: Optional[str] = None,
    life_stage: Optional[str] = None,
    topic: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
) -> List[Document]:
    """根据 metadata 做简单过滤，不足再用相似度结果补齐"""
    res = []
    for d in docs:
        m = d.metadata or {}
        ok = True
        if species:
            ok &= (m.get("species") == species)
        if breed:
            ok &= (m.get("breed") == breed)
        if life_stage:
            ok &= (m.get("life_stage") == life_stage)
        if topic:
            ok &= (m.get("topic") == topic)
        if ok:
            res.append(d)
        if len(res) >= top_k:
            break

    if len(res) < top_k:
        need = top_k - len(res)
        for d in docs:
            if d not in res:
                res.append(d)
                if len(res) >= top_k:
                    break
    return res[:top_k]


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source_name", "source")
        url = meta.get("source_url", "")
        date = meta.get("source_date") or meta.get("crawl_date") or ""
        header = f"[{i}] {src} ({url}) {date}".strip()
        blocks.append(header + "\n" + d.page_content.strip())
    return "\n\n---\n\n".join(blocks)


def call_openai(question: str, retrieved_docs: List[Document]):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set. 请先设置环境变量。")

    context = format_context(retrieved_docs)
    system_prompt = (
        "You are a veterinary-informed pet care assistant for cats and dogs.\n"
        "Rules:\n"
        "- Only use the provided context; if information is missing, say you're unsure and recommend seeing a veterinarian.\n"
        "- Be specific by species/life stage if available. Use clear bullet points.\n"
        "- Always include inline citations like [1], [2] referring to the context block indices.\n"
        "- Add a brief safety note if advice relates to medication/supplements.\n"
        "- Neutral, educational tone.\n"
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Context blocks (indexed):\n{context}\n\n"
        "Now write a helpful answer with bullets and citations."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", None),
        "output_tokens": getattr(resp.usage, "completion_tokens", None),
        "total_tokens": getattr(resp.usage, "total_tokens", None),
    }
    return answer, usage



def retrieve(db: Chroma, query: str, k: int = TOP_K_DEFAULT) -> List[Document]:
    # 先取 20 个，再用 metadata 过滤
    return db.similarity_search(query, k=20)


# ============= FastAPI 模型 & 路由 =============

class AskBody(BaseModel):
    question: str
    species: Optional[str] = None
    breed: Optional[str] = None
    life_stage: Optional[str] = None
    topic: Optional[str] = None
    top_k: int = TOP_K_DEFAULT
    eval_id: Optional[str] = None 



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_api(body: AskBody):
    """
    Main RAG entrypoint. Any exception will be returned to the caller
    with its message so the frontend/curl can see root cause.
    """
    import traceback

    global vectordb
    start = time.time()
    success = True
    answer = None
    usage = None
    filtered_docs: List[Document] = []

    try:
        # lazy-load vector DB inside try so load errors are surfaced
        if vectordb is None:
            vectordb = load_vectordb()

        raw_docs = retrieve(vectordb, body.question, k=body.top_k)
        filtered_docs = filter_by_metadata(
            raw_docs,
            species=body.species,
            breed=body.breed,
            life_stage=body.life_stage,
            topic=body.topic,
            top_k=body.top_k,
        )
        answer, usage = call_openai(body.question, filtered_docs)
        response = {
            "answer": answer,
            "matches": [
                {"text": d.page_content, "metadata": d.metadata}
                for d in filtered_docs
            ],
            "usage": usage,
        }
    except Exception as e:
        success = False
        trace = traceback.format_exc()
        print("ERROR /ask:", e, "\n", trace)
        response = {"error": str(e), "trace": trace}
    finally:
        latency = time.time() - start
        try:
            requests.post(
                "http://localhost:8010/log",
                json={
                    "endpoint": "/ask",
                    "latency": latency,
                    "success": success,
                    "question": body.question,
                    "answer": answer,
                    "retrieved_docs": [d.page_content for d in filtered_docs],
                    "usage": usage,
                    "eval_id": body.eval_id,
                },
                timeout=0.5,
            )
        except Exception:
            pass

    return response


# ============= CLI 入口 =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build vector DB from ./data JSONL files")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.build:
        build_index()

    if args.serve:
        load_vectordb()
        print(f"🚀 Serving on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    if not args.build and not args.serve:
        parser.print_help()


if __name__ == "__main__":
    main()
