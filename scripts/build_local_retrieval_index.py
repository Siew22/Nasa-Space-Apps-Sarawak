from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
from pathlib import Path
import numpy as np

from app.settings import find_project_root, read_ai_config

ROOT = find_project_root()
CORPUS_FILE = ROOT / "data" / "knowledge" / "nasa_combustion_corpus.json"
INDEX_DIR = ROOT / "data" / "vector_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

PASSAGES_PATH = INDEX_DIR / "passages.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
INDEX_META_PATH = INDEX_DIR / "index_metadata.json"


def build_index() -> None:
    from sentence_transformers import SentenceTransformer
    import faiss

    cfg = read_ai_config()
    embed_local_dir = ROOT / cfg["embedding_model"]["local_path"]

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"语料库文件缺失: {CORPUS_FILE}")

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    passages = []
    texts_to_embed = []

    for doc in corpus["documents"]:
        doc_id = doc["doc_id"]
        citation = doc["citation"]
        doi = doc.get("doi")
        inv = doc["investigation"]

        for p in doc["passages"]:
            passage_entry = {
                "passage_id": p["passage_id"],
                "doc_id": doc_id,
                "investigation": inv,
                "citation": citation,
                "doi": doi,
                "ntrs_id": doc.get("ntrs_id"),
                "physical_system": p["physical_system"],
                "text": p["text"],
                "conditions": p["conditions"],
            }
            passages.append(passage_entry)
            texts_to_embed.append(f"{inv} ({citation}) {p['topic']}: {p['text']}")

    print(f"📦 正在提取并向量化 {len(passages)} 个可追溯科学分块...")

    # 优先使用本地固化的嵌入模型，若未下载则提示
    if embed_local_dir.exists() and any(embed_local_dir.iterdir()):
        embed_model = SentenceTransformer(str(embed_local_dir))
    else:
        embed_model = SentenceTransformer(cfg["embedding_model"]["model_name"])

    # 关键步骤：强制 L2 归一化，使得内积在数学上严格等于余弦相似度
    embeddings = embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    # 显式构建 IndexFlatIP (Inner Product)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2, ensure_ascii=False)

    # 固化索引元数据，消除度量歧义
    index_meta = {
        "index_type": "IndexFlatIP",
        "metric": "cosine_similarity_via_inner_product",
        "is_l2_normalized": True,
        "embedding_dimension": dim,
        "total_passages": len(passages),
        "similarity_score_range": "[-1.0, 1.0]; values > 0.35 indicate semantic relevance"
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, indent=2, ensure_ascii=False)

    print(f"✅ FAISS 本地索引已构建: {FAISS_INDEX_PATH} (度量: IndexFlatIP)")
    print(f"✅ 索引度量说明文件已落盘: {INDEX_META_PATH}")


if __name__ == "__main__":
    build_index()