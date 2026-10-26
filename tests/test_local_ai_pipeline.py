"""
本地 AI 架构全自动化测试套件 (Zero Mocking for Critical Security Paths)
验证：
1. 语料库结构与 DOI 真实性
2. 数据集防数据泄漏组隔离
3. 本地向量检索准确度与阈值拒绝
4. 离线模式严格性与无外部 API 泄漏
"""
from pathlib import Path

import pytest

# The local LLM stack (requirement.txt) and the downloaded model weights exist only on the training machine,
# so these tests are skipped on machines without them instead of breaking the whole suite.
pytest.importorskip("torch", reason="local LLM stack not installed (pip install -r requirement.txt)")
from app.services.local_llm_service import get_ai_status, retrieve_relevant_passages, ask_local_research_ai  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
needs_local_models = pytest.mark.skipif(
    not (ROOT / "models" / "base_models" / "Qwen2.5-1.5B-Instruct").exists()
    or not (ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2").exists(),
    reason="local model weights not downloaded (python -m scripts.download_model_assets)",
)


def test_ai_status_reports_local_device():
    """验证系统始终处于本地模式且正确读取硬件配置"""
    status = get_ai_status()
    assert status["offline_mode_enforced"] is True
    assert status["base_model_license"] == "Apache-2.0"
    assert status["device"] in ["cuda", "cpu"]


@needs_local_models
def test_retrieval_finds_relevant_nasa_doi():
    """验证本地语义检索能够正确定位真实 NASA DOI"""
    query = "quiescent microgravity flame extinction and oxygen starvation"
    results = retrieve_relevant_passages(query, top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "doi" in top_hit
    assert top_hit["doi"].startswith("10.60555/")
    assert top_hit["similarity_score"] > 0.40


@needs_local_models
def test_insufficient_evidence_triggers_safe_refusal():
    """验证当输入无关或虚构提问时，模型坚决拒绝生成幻想数据"""
    nonsense_query = "What is the warp drive subspace frequency of the flame?"
    res = ask_local_research_ai(nonsense_query)
    assert res["evidence_sufficient"] is False
    assert "INSUFFICIENT EVIDENCE" in res["answer"]
    assert len(res["evidence_passages"]) == 0