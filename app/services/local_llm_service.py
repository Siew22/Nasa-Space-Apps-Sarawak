from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
from typing import Any
import torch

from app.settings import find_project_root, read_ai_config

ROOT = find_project_root()
PASSAGES_PATH = ROOT / "data" / "vector_index" / "passages.json"
FAISS_INDEX_PATH = ROOT / "data" / "vector_index" / "faiss_index.bin"

_TOKENIZER = None
_MODEL = None
_EMBED_MODEL = None
_FAISS_INDEX = None
_PASSAGES = None
_HAS_ADAPTER = False


def get_ai_status() -> dict[str, Any]:
    cfg = read_ai_config()
    local_base = ROOT / cfg["base_model"]["local_path"]
    adapter_path = ROOT / cfg["adapter_output_dir"]
    has_base = local_base.exists() and any(local_base.iterdir())
    has_adapter = (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()
    has_index = FAISS_INDEX_PATH.exists() and PASSAGES_PATH.exists()

    return {
        "service_status": "online_local",
        "offline_mode_configured": cfg.get("strict_offline_mode", True),
        # Keys read by the SI tab in src/app.py and by tests/test_local_ai_pipeline.py
        "offline_mode_enforced": bool(cfg.get("strict_offline_mode", True)) and os.environ.get("HF_HUB_OFFLINE") == "1",
        "base_model": cfg["base_model"]["model_name"],
        "base_model_license": cfg["base_model"].get("license"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "base_model_path": str(local_base),
        "local_base_model_available": has_base,
        "lora_adapter_available": has_adapter,
        "lora_adapter_path": str(adapter_path) if has_adapter else None,
        "retrieval_index_available": has_index,
        "active_model_mode": "Fine-Tuned Space LoRA Model" if has_adapter else ("Pre-trained Base Model (Zero-Shot RAG)" if has_base else "Model Weights Not Downloaded")
    }


def _lazy_init():
    global _TOKENIZER, _MODEL, _EMBED_MODEL, _FAISS_INDEX, _PASSAGES, _HAS_ADAPTER

    if _TOKENIZER is not None:
        return

    cfg = read_ai_config()
    from sentence_transformers import SentenceTransformer
    import faiss
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    embed_dir = ROOT / cfg["embedding_model"]["local_path"]
    if embed_dir.exists() and FAISS_INDEX_PATH.exists() and PASSAGES_PATH.exists():
        _EMBED_MODEL = SentenceTransformer(str(embed_dir))
        _FAISS_INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
            _PASSAGES = json.load(f)

    local_base = ROOT / cfg["base_model"]["local_path"]
    _TOKENIZER = AutoTokenizer.from_pretrained(str(local_base), local_files_only=True)

    base = AutoModelForCausalLM.from_pretrained(
        str(local_base),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        local_files_only=True
    )

    adapter_dir = ROOT / cfg["adapter_output_dir"]
    has_weights = (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists()

    if has_weights:
        _MODEL = PeftModel.from_pretrained(base, str(adapter_dir))
        _HAS_ADAPTER = True
    else:
        _MODEL = base
        _HAS_ADAPTER = False


def retrieve_relevant_passages(query: str, top_k: int | None = None, threshold: float | None = None) -> list[dict[str, Any]]:
    _lazy_init()
    if _FAISS_INDEX is None or _PASSAGES is None:
        return []

    cfg = read_ai_config()
    k = top_k or cfg["retrieval"].get("top_k", 3)
    limit = threshold if threshold is not None else cfg["retrieval"].get("similarity_threshold", 0.35)

    import numpy as np
    q_vec = _EMBED_MODEL.encode([query], normalize_embeddings=True)
    D, I = _FAISS_INDEX.search(np.ascontiguousarray(q_vec, dtype=np.float32), k=k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_PASSAGES):
            continue
        if float(score) >= limit:
            item = dict(_PASSAGES[idx])
            item["similarity_score"] = round(float(score), 4)
            results.append(item)

    return results


def ask_local_research_ai(question: str, use_fine_tuned: bool = True) -> dict[str, Any]:
    status = get_ai_status()
    if not status["local_base_model_available"]:
        return {
            "question": question,
            "answer": "SYSTEM NOTICE: Local base model weights are not downloaded yet.",
            "evidence_passages": [],
            "evidence_sufficient": False,
            "model_metadata": status
        }

    if use_fine_tuned and not status["lora_adapter_available"]:
        raise FileNotFoundError(
            "无法执行微调模型推理：LoRA 适配器权重文件不存在。请先运行 'python -m scripts.train_local_llm'。"
        )

    _lazy_init()
    passages = retrieve_relevant_passages(question)
    # Supporting passages only need similarity_threshold, but answering at all needs one strong match: an off-topic
    # question sharing a word like "flame" scores ~0.35, the sample and benchmark questions 0.42 or more.
    min_top = read_ai_config()["retrieval"].get("min_top_similarity_to_answer", 0.0)

    if not passages or max(p["similarity_score"] for p in passages) < min_top:
        return {
            "question": question,
            "answer": "INSUFFICIENT EVIDENCE: The local NASA research database does not contain sufficient verified experimental passages matching this query.",
            "evidence_passages": [],
            "evidence_sufficient": False,
            "model_metadata": status
        }

    context_str = "\n\n".join([f"[{p['citation']} | DOI: {p.get('doi') or 'None'}] {p['text']}" for p in passages])
    system_prompt = (
        "You are the NASA Microgravity Fire Safety Research Intelligence Engine (Flame in Freefall SI). "
        "Answer the user's question using ONLY the provided NASA evidence passages. Cite the document ID and DOI. "
        "Explicitly distinguish solid fuel surface flame spread (BASS-II/Saffire) from liquid droplet combustion (FLEX). "
        "Do not invent facts."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Supporting Evidence:\n{context_str}\n\nQuestion:\n{question}"}
    ]

    prompt = _TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)

    def _generate(model_to_use):
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=_TOKENIZER.eos_token_id
            )
        return _TOKENIZER.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if _HAS_ADAPTER and not use_fine_tuned:
        with _MODEL.disable_adapter():
            response_text = _generate(_MODEL)
    else:
        response_text = _generate(_MODEL)

    return {
        "question": question,
        "answer": response_text,
        "evidence_passages": passages,
        "evidence_sufficient": True,
        "model_used": "Fine-Tuned Adapter" if (use_fine_tuned and _HAS_ADAPTER) else "Unmodified Base Model (via disable_adapter)",
        "model_metadata": status
    }
