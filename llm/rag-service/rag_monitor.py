"""
rag_monitor.py
Pawty RAG Monitoring & Evaluation (Enhanced)

Usage:
  # 实时监控服务
  python rag_monitor.py --serve --port 8010

  # 离线评估（可选）
  python rag_monitor.py --eval data/eval_set.json --api http://localhost:8001/ask
"""

import json
import time
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import requests

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt", quiet=True)

# ===== 工具函数：评估 =====

def evaluate_retrieval(predicted_docs: List[str], true_docs: List[str]):
    """简单字符串包含匹配，返回 (precision, recall)"""
    if not predicted_docs or not true_docs:
        return 0.0, 0.0

    hits = 0
    for td in true_docs:
        td_stripped = td.strip()
        if not td_stripped:
            continue
        for pd in predicted_docs:
            if td_stripped in pd:
                hits += 1
                break

    precision = hits / len(predicted_docs)
    recall = hits / len(true_docs)
    return precision, recall


def evaluate_generation(pred_answer: str, ref_answer: str) -> float:
    if not pred_answer or not ref_answer:
        return 0.0
    return sentence_bleu([ref_answer.split()], pred_answer.split())


def estimate_hallucination_risk(answer: str, retrieved_docs: List[str]) -> float:
    """
    简单启发式：
    - 统计 answer 中的词
    - 看有多少词从未出现在 retrieved_docs 中
    - 比例越高，幻觉风险越高
    """
    import re

    if not answer or not retrieved_docs:
        return 0.0

    doc_text = " ".join(retrieved_docs).lower()
    doc_tokens = set(re.findall(r"\w+", doc_text))
    ans_tokens = re.findall(r"\w+", answer.lower())

    if not ans_tokens:
        return 0.0

    novel = [t for t in ans_tokens if t not in doc_tokens]
    return len(novel) / len(ans_tokens)


# ===== 离线评估模式 =====

def run_evaluation(eval_path: str, api_url: str = "http://localhost:8001/ask"):
    print(f"🔍 Running offline eval with {eval_path} ...")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data: List[Dict[str, Any]] = json.load(f)

    precisions, recalls, bleus = [], [], []

    for item in eval_data:
        q = item["question"]
        ref = item["reference_answer"]
        true_docs = item.get("relevant_docs", [])

        body = {"question": q, "eval_id": item.get("eval_id")}
        t0 = time.time()
        resp = requests.post(api_url, json=body, timeout=60)
        dt = time.time() - t0

        if resp.status_code != 200:
            print(f"[WARN] Request failed: {resp.status_code}, {resp.text}")
            continue

        res = resp.json()
        pred_answer = res.get("answer", "")
        predicted_docs = [m["text"] for m in res.get("matches", [])]

        p, r = evaluate_retrieval(predicted_docs, true_docs)
        b = evaluate_generation(pred_answer, ref)

        precisions.append(p)
        recalls.append(r)
        bleus.append(b)

        print(f"Q: {q}")
        print(f"  latency: {dt:.2f}s, P={p:.2f}, R={r:.2f}, BLEU={b:.2f}")

    if not precisions:
        print("❗ No successful eval samples.")
        return

    print("\n===== Eval Summary =====")
    print(f"📊 Precision: {np.mean(precisions):.3f}")
    print(f"📈 Recall:    {np.mean(recalls):.3f}")
    print(f"🧠 BLEU:      {np.mean(bleus):.3f}")


# ===== 实时监控 API =====

app = FastAPI(title="Pawty RAG Monitor", version="2.0.0")

# 所有日志（简单起见存在内存里）
logs: List[Dict[str, Any]] = []

# 题库（eval_set），用于在实时监控时计算 PR / BLEU
eval_index_by_q: Dict[str, Dict[str, Any]] = {}
eval_index_by_id: Dict[str, Dict[str, Any]] = {}


def load_eval_set(path: str = "data/eval_set.json"):
    global eval_index_by_q, eval_index_by_id
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ℹ️ No eval_set.json found at {path}. Real-time PR/Recall/BLEU will be 0.")
        return

    for item in data:
        q = item.get("question")
        eid = item.get("eval_id")
        if q:
            eval_index_by_q[q] = item
        if eid:
            eval_index_by_id[eid] = item

    print(f"✅ Loaded eval set: {len(data)} items.")


class TokenUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class Log(BaseModel):
    endpoint: str
    latency: float
    success: bool
    question: Optional[str] = None
    answer: Optional[str] = None
    retrieved_docs: Optional[List[str]] = None
    usage: Optional[TokenUsage] = None
    eval_id: Optional[str] = None


@app.post("/log")
def log_request(entry: Log):
    logs.append(entry.dict())
    return {"message": "logged"}


@app.get("/metrics")
def get_metrics():
    if not logs:
        return {"message": "no logs yet"}

    total = len(logs)
    succ = [l for l in logs if l["success"]]
    success_rate = len(succ) / total
    latencies = [l["latency"] for l in logs]

    # token stats
    input_tokens = []
    output_tokens = []
    total_tokens = []
    for l in logs:
        u = l.get("usage") or {}
        if u.get("input_tokens") is not None:
            input_tokens.append(u["input_tokens"])
        if u.get("output_tokens") is not None:
            output_tokens.append(u["output_tokens"])
        if u.get("total_tokens") is not None:
            total_tokens.append(u["total_tokens"])

    # quality metrics (only for logs that match eval_set)
    prs, rcs, bleus, halluc_scores = [], [], [], []
    for l in logs:
        q = l.get("question")
        eid = l.get("eval_id")
        answer = l.get("answer") or ""
        docs = l.get("retrieved_docs") or []

        gt = None
        if eid and eid in eval_index_by_id:
            gt = eval_index_by_id[eid]
        elif q and q in eval_index_by_q:
            gt = eval_index_by_q[q]

        if gt:
            true_docs = gt.get("relevant_docs", [])
            ref_answer = gt.get("reference_answer", "")

            p, r = evaluate_retrieval(docs, true_docs)
            b = evaluate_generation(answer, ref_answer)
            h = estimate_hallucination_risk(answer, docs)

            prs.append(p)
            rcs.append(r)
            bleus.append(b)
            halluc_scores.append(h)

    metrics = {
        "total_requests": total,
        "success_rate": round(success_rate, 3),
        "avg_latency": round(float(np.mean(latencies)), 3),
        "avg_input_tokens": float(np.mean(input_tokens)) if input_tokens else None,
        "avg_output_tokens": float(np.mean(output_tokens)) if output_tokens else None,
        "avg_total_tokens": float(np.mean(total_tokens)) if total_tokens else None,
        "avg_answer_length_chars": float(
            np.mean([len(l.get("answer") or "") for l in logs])
        ),
    }

    if prs:
        metrics.update(
            dict(
                retrieval_precision=float(np.mean(prs)),
                retrieval_recall=float(np.mean(rcs)),
                bleu=float(np.mean(bleus)),
                hallucination_risk=float(np.mean(halluc_scores)),
                eval_sample_count=len(prs),
            )
        )
    else:
        metrics.update(
            dict(
                retrieval_precision=None,
                retrieval_recall=None,
                bleu=None,
                hallucination_risk=None,
                eval_sample_count=0,
            )
        )

    return metrics


@app.get("/health")
def health():
    return {"status": "ok", "log_count": len(logs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Start monitoring API server")
    parser.add_argument("--eval", type=str, help="Path to eval JSON file")
    parser.add_argument("--api", type=str, default="http://localhost:8001/ask", help="RAG /ask endpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    if args.eval:
        run_evaluation(args.eval, api_url=args.api)

    if args.serve:
        load_eval_set()   # 实时监控时尝试加载 eval_set.json
        print(f"🚀 Starting monitor on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    if not args.eval and not args.serve:
        parser.print_help()


if __name__ == "__main__":
    main()
