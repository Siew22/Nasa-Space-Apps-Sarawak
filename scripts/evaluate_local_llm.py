from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
import time
from pathlib import Path
import pandas as pd
from app.settings import find_project_root
from app.services.local_llm_service import ask_local_research_ai, get_ai_status

ROOT = find_project_root()
TEST_FILE = ROOT / "data" / "ai_training" / "test.jsonl"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model_pipeline(use_fine_tuned: bool, model_name: str, test_examples: list[dict]) -> tuple[dict, list[dict]]:
    status = get_ai_status()
    if use_fine_tuned and not status["lora_adapter_available"]:
        raise FileNotFoundError(
            f"无法评测微调模型：未在 {status['lora_adapter_path']} 找到有效的 LoRA 权重！"
            "请先成功运行 'python -m scripts.train_local_llm'。"
        )

    print(f"\n🔬 正在启动模型独立基准测试: 【{model_name}】...")
    
    # 1. 显式冷启动预热 (排除模型加载与 PyTorch 编译对推理延迟统计的污染)
    t_warm_start = time.time()
    _ = ask_local_research_ai("Warmup query: what is microgravity combustion?", use_fine_tuned=use_fine_tuned)
    cold_start_latency = round(time.time() - t_warm_start, 2)
    print(f"   -> 冷启动初始化耗时: {cold_start_latency} s (不计入稳态评测均值)")

    records = []
    steady_latencies = []
    tp, fp, tn, fn = 0, 0, 0, 0
    citation_string_hits = 0
    substantiated_queries_count = 0

    for ex in test_examples:
        q = ex["input"]
        expected_refusal = (ex["task_type"] == "insufficient_evidence_refusal")
        expected_doi = ex.get("doi")
        expected_citation = ex.get("citation")

        t0 = time.time()
        res = ask_local_research_ai(q, use_fine_tuned=use_fine_tuned)
        lat = time.time() - t0
        steady_latencies.append(lat)

        ans = res["answer"]
        model_refused = ("INSUFFICIENT EVIDENCE" in ans)

        # 拒答行为混淆矩阵统计
        if expected_refusal and model_refused:
            tp += 1
        elif not expected_refusal and model_refused:
            fp += 1
        elif not expected_refusal and not model_refused:
            tn += 1
        elif expected_refusal and not model_refused:
            fn += 1

        # 严格引文字符串匹配 (仅对有明确来源凭据的问题计算，排除拒答题)
        matched_expected_string = False
        if not expected_refusal:
            substantiated_queries_count += 1
            if (expected_doi and str(expected_doi).strip() and str(expected_doi) in ans) or \
               (expected_citation and str(expected_citation).strip() and str(expected_citation) in ans):
                matched_expected_string = True
                citation_string_hits += 1

        records.append({
            "model_tested": model_name,
            "example_id": ex["example_id"],
            "task_type": ex["task_type"],
            "expected_refusal": expected_refusal,
            "model_refused": model_refused,
            "exact_citation_string_hit": matched_expected_string,
            "steady_state_latency_s": round(lat, 2)
        })

    # 计算科学指标
    refusal_precision = round(tp / max(1, (tp + fp)), 4)
    refusal_recall = round(tp / max(1, (tp + fn)), 4)
    refusal_f1 = round(2 * tp / max(1, (2 * tp + fp + fn)), 4)
    citation_match_rate = round(citation_string_hits / max(1, substantiated_queries_count), 4)
    avg_steady_latency = round(sum(steady_latencies) / max(1, len(steady_latencies)), 2)

    summary = {
        "model_name": model_name,
        "evaluated_examples_count": len(test_examples),
        "cold_start_latency_s": cold_start_latency,
        "mean_steady_state_latency_s": avg_steady_latency,
        "exact_citation_string_match_rate": citation_match_rate,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1_score": refusal_f1,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "scientific_disclaimer": "exact_citation_string_match_rate measures attribution format compliance, not exhaustive physical truthfulness."
    }

    return summary, records


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"盲测集缺失: {TEST_FILE}")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]

    # 1. 评测纯基座模型 (通过 disable_adapter 保证零权重污染)
    base_summary, base_records = evaluate_model_pipeline(
        use_fine_tuned=False,
        model_name="Unmodified Base Model (Zero-Shot RAG)",
        test_examples=test_examples
    )

    # 2. 评测微调模型 (必须存在 Adapter 否则直接抛出异常)
    lora_summary, lora_records = evaluate_model_pipeline(
        use_fine_tuned=True,
        model_name="Fine-Tuned Space LoRA Model",
        test_examples=test_examples
    )

    comparison_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset_file": str(TEST_FILE),
        "test_examples_count": len(test_examples),
        "comparative_metrics": [base_summary, lora_summary]
    }

    json_path = REPORTS_DIR / "local_ai_comparative_benchmark.json"
    csv_path = REPORTS_DIR / "local_ai_comparative_benchmark.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)

    pd.DataFrame(base_records + lora_records).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("🏆 【基座模型 vs 微调模型】独立盲测对比基准报告 (Side-by-Side Comparison):")
    display_df = pd.DataFrame([base_summary, lora_summary])[
        ["model_name", "exact_citation_string_match_rate", "refusal_f1_score", "mean_steady_state_latency_s"]
    ]
    print(display_df.to_string(index=False))
    print("=" * 75)
    print(f"📁 详细评测报告已落盘至: {json_path}")
    print(f"📁 逐样本命中结果已落盘至: {csv_path}\n")


if __name__ == "__main__":
    main()