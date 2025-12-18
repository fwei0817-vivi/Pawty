"""
rag_monitor.py
Pawty RAG Monitoring & Evaluation (Enhanced)

Usage:
  # Real-time monitoring service
  python rag_monitor.py --serve --port 8010

  # Offline evaluation (optional)
  python rag_monitor.py --eval data/eval_set.json --api http://localhost:8001/ask
"""

import json
import time
import argparse
import os
from typing import List, Dict, Any, Optional

import numpy as np
import requests

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import nltk
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
from dotenv import load_dotenv

nltk.download("punkt", quiet=True)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_judge_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Evaluation Utility Functions =====

def evaluate_retrieval(predicted_docs: List[str], true_docs: List[str]):
    """Simple string containment matching, returns (precision, recall)"""
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
    Simple heuristic:
    - Count words in answer
    - Count words that never appear in retrieved_docs
    - Higher ratio indicates higher hallucination risk
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


def calculate_hit_at_k(predicted_docs: List[str], true_docs: List[str], k: int = 5) -> float:
    """
    Hit@k: Check if at least one relevant document is in the top k retrieval results
    Returns 1.0 if found, otherwise 0.0
    """
    if not predicted_docs or not true_docs:
        return 0.0
    
    top_k_docs = predicted_docs[:k]
    for td in true_docs:
        td_stripped = td.strip()
        if not td_stripped:
            continue
        for pd in top_k_docs:
            if td_stripped in pd:
                return 1.0
    return 0.0


def calculate_mrr(predicted_docs: List[str], true_docs: List[str]) -> float:
    """
    MRR (Mean Reciprocal Rank): Calculate reciprocal of the rank of the first relevant document
    If the first relevant document is at rank position, MRR = 1/rank
    Returns 0.0 if not found
    """
    if not predicted_docs or not true_docs:
        return 0.0
    
    for rank, pd in enumerate(predicted_docs, start=1):
        for td in true_docs:
            td_stripped = td.strip()
            if td_stripped and td_stripped in pd:
                return 1.0 / rank
    return 0.0


def llm_judge_faithfulness(answer: str, retrieved_docs: List[str]) -> Optional[float]:
    """
    Answer Faithfulness (LLM-as-judge): 
    Judge if answer is faithful to retrieved documents (no hallucinations)
    Returns score between 0.0-1.0, None if cannot judge
    """
    if not llm_judge_client or not answer or not retrieved_docs:
        return None
    
    context = "\n\n".join([f"[Doc {i+1}]: {doc[:500]}" for i, doc in enumerate(retrieved_docs[:3])])
    
    prompt = f"""You are evaluating whether an answer is faithful to the provided context documents.

Context Documents:
{context}

Answer to evaluate:
{answer}

Please evaluate if the answer is fully supported by the context documents. Consider:
1. Are all factual claims in the answer directly supported by the context?
2. Does the answer contain any information not present in the context?
3. Are there any contradictions between the answer and the context?

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}

Score interpretation:
- 1.0: Answer is completely faithful, all claims are supported by context
- 0.7-0.9: Answer is mostly faithful with minor unsupported details
- 0.4-0.6: Answer has some unsupported claims or contradictions
- 0.0-0.3: Answer contains significant unsupported information or contradictions
"""
    
    try:
        response = llm_judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0))
    except Exception as e:
        print(f"[WARN] LLM judge faithfulness failed: {e}")
        return None


def llm_judge_relevance(question: str, answer: str) -> Optional[float]:
    """
    Answer Relevance (LLM-as-judge):
    Judge if answer is relevant to the question
    Returns score between 0.0-1.0, None if cannot judge
    """
    if not llm_judge_client or not question or not answer:
        return None
    
    prompt = f"""You are evaluating whether an answer is relevant to the given question.

Question:
{question}

Answer:
{answer}

Please evaluate if the answer directly addresses the question. Consider:
1. Does the answer provide information that directly relates to the question?
2. Is the answer helpful in answering the question?
3. Does the answer stay on topic?

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}

Score interpretation:
- 1.0: Answer is highly relevant and directly addresses the question
- 0.7-0.9: Answer is mostly relevant with minor off-topic content
- 0.4-0.6: Answer is partially relevant but misses key aspects
- 0.0-0.3: Answer is largely irrelevant or off-topic
"""
    
    try:
        response = llm_judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0))
    except Exception as e:
        print(f"[WARN] LLM judge relevance failed: {e}")
        return None


# ===== Offline Evaluation Mode =====

def run_evaluation(eval_path: str, api_url: str = "http://localhost:8001/ask"):
    print(f"🔍 Running offline eval with {eval_path} ...")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data: List[Dict[str, Any]] = json.load(f)

    precisions, recalls, hit_at_1s, hit_at_3s, hit_at_5s, mrrs = [], [], [], [], [], []
    faithfulness_scores, relevance_scores = [], []

    for item in eval_data:
        q = item["question"]
        ref = item.get("reference_answer", "")
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

        # Traditional metrics
        p, r = evaluate_retrieval(predicted_docs, true_docs)
        hit_at_1 = calculate_hit_at_k(predicted_docs, true_docs, k=1)
        hit_at_3 = calculate_hit_at_k(predicted_docs, true_docs, k=3)
        hit_at_5 = calculate_hit_at_k(predicted_docs, true_docs, k=5)
        mrr = calculate_mrr(predicted_docs, true_docs)

        precisions.append(p)
        recalls.append(r)
        hit_at_1s.append(hit_at_1)
        hit_at_3s.append(hit_at_3)
        hit_at_5s.append(hit_at_5)
        mrrs.append(mrr)

        # LLM-as-judge metrics
        faithfulness = llm_judge_faithfulness(pred_answer, predicted_docs)
        relevance = llm_judge_relevance(q, pred_answer)
        
        if faithfulness is not None:
            faithfulness_scores.append(faithfulness)
        if relevance is not None:
            relevance_scores.append(relevance)

        print(f"Q: {q}")
        print(f"  latency: {dt:.2f}s, P={p:.2f}, R={r:.2f}, Hit@1={hit_at_1:.2f}, Hit@3={hit_at_3:.2f}, Hit@5={hit_at_5:.2f}, MRR={mrr:.3f}")
        if faithfulness is not None:
            print(f"  Faithfulness: {faithfulness:.3f}")
        if relevance is not None:
            print(f"  Relevance: {relevance:.3f}")

    if not precisions:
        print("❗ No successful eval samples.")
        return

    print("\n===== Eval Summary =====")
    print(f"📊 Precision:     {np.mean(precisions):.3f}")
    print(f"📈 Recall:        {np.mean(recalls):.3f}")
    print(f"🎯 Hit@1:         {np.mean(hit_at_1s):.3f}")
    print(f"🎯 Hit@3:         {np.mean(hit_at_3s):.3f}")
    print(f"🎯 Hit@5:         {np.mean(hit_at_5s):.3f}")
    print(f"📉 MRR:           {np.mean(mrrs):.3f}")
    if faithfulness_scores:
        print(f"✅ Faithfulness:  {np.mean(faithfulness_scores):.3f} (n={len(faithfulness_scores)})")
    if relevance_scores:
        print(f"🔗 Relevance:     {np.mean(relevance_scores):.3f} (n={len(relevance_scores)})")


# ===== Real-time Monitoring API =====

app = FastAPI(title="Pawty RAG Monitor", version="2.0.0")

# All logs (stored in memory for simplicity)
logs: List[Dict[str, Any]] = []

# Evaluation set (eval_set), used for computing PR/BLEU in real-time monitoring
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
    prs, rcs, hit_at_1s, hit_at_3s, hit_at_5s, mrrs = [], [], [], [], [], []
    faithfulness_scores, relevance_scores = [], []
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

            p, r = evaluate_retrieval(docs, true_docs)
            hit_at_1 = calculate_hit_at_k(docs, true_docs, k=1)
            hit_at_3 = calculate_hit_at_k(docs, true_docs, k=3)
            hit_at_5 = calculate_hit_at_k(docs, true_docs, k=5)
            mrr = calculate_mrr(docs, true_docs)

            prs.append(p)
            rcs.append(r)
            hit_at_1s.append(hit_at_1)
            hit_at_3s.append(hit_at_3)
            hit_at_5s.append(hit_at_5)
            mrrs.append(mrr)

            # LLM-as-judge metrics (only compute if we have OpenAI key)
            if llm_judge_client:
                faithfulness = llm_judge_faithfulness(answer, docs)
                relevance = llm_judge_relevance(q, answer)
                if faithfulness is not None:
                    faithfulness_scores.append(faithfulness)
                if relevance is not None:
                    relevance_scores.append(relevance)

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
                hit_at_1=float(np.mean(hit_at_1s)),
                hit_at_3=float(np.mean(hit_at_3s)),
                hit_at_5=float(np.mean(hit_at_5s)),
                mrr=float(np.mean(mrrs)),
                eval_sample_count=len(prs),
            )
        )
        if faithfulness_scores:
            metrics["answer_faithfulness"] = float(np.mean(faithfulness_scores))
        if relevance_scores:
            metrics["answer_relevance"] = float(np.mean(relevance_scores))
    else:
        metrics.update(
            dict(
                retrieval_precision=None,
                retrieval_recall=None,
                hit_at_1=None,
                hit_at_3=None,
                hit_at_5=None,
                mrr=None,
                answer_faithfulness=None,
                answer_relevance=None,
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
        load_eval_set()   # Try to load eval_set.json for real-time monitoring
        print(f"🚀 Starting monitor on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    if not args.eval and not args.serve:
        parser.print_help()


if __name__ == "__main__":
    main()
