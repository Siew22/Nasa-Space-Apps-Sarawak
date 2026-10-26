from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise RuntimeError("硬件环境不满足要求！必须在具备 NVIDIA GPU (CUDA) 的环境下执行。")

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from app.settings import find_project_root, read_ai_config

ROOT = find_project_root()
TRAIN_FILE = ROOT / "data" / "ai_training" / "train.jsonl"
VAL_FILE = ROOT / "data" / "ai_training" / "validation.jsonl"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def tokenize_with_budget(example: dict, tokenizer, max_seq_length: int = 768) -> dict:
    system_text = (
        "<|im_start|>system\n"
        "You are the NASA Microgravity Fire Safety Research Intelligence Engine. "
        "Answer questions based strictly on the provided NASA evidence passages. "
        "If evidence is insufficient, explicitly state INSUFFICIENT EVIDENCE.<|im_end|>\n"
    )
    question_text = f"\n\nQuestion:\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    target_text = f"{example['target']}<|im_end|>"

    system_ids = tokenizer.encode(system_text, add_special_tokens=False)
    question_ids = tokenizer.encode(question_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    reserved_len = len(system_ids) + len(question_ids) + len(target_ids)
    context_budget = max_seq_length - reserved_len

    if context_budget <= 20:
        raise ValueError(f"样本 {example.get('example_id')} 问答对本身过长！")

    context_prefix = "<|im_start|>user\nContext:\n"
    context_ids = tokenizer.encode(context_prefix + example["context"], add_special_tokens=False)

    was_truncated = False
    if len(context_ids) > context_budget:
        context_ids = context_ids[:context_budget]
        was_truncated = True

    prompt_ids = system_ids + context_ids + question_ids
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if not any(lbl != -100 for lbl in labels):
        raise AssertionError(f"样本 {example.get('example_id')} 无有效训练标签！")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "was_truncated": was_truncated
    }


def main():
    cfg = read_ai_config()
    local_base_path = ROOT / cfg["base_model"]["local_path"]
    out_dir = ROOT / cfg["adapter_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"🚀 [Native {'BF16' if use_bf16 else 'FP16'} LoRA] 显卡: {torch.cuda.get_device_name(0)}")
    print(f"📦 载入基座模型 (精度模式: {model_dtype}): {local_base_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(local_base_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(local_base_path),
        torch_dtype=model_dtype,
        device_map="auto",
        local_files_only=True
    )
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=cfg["training"]["lora_r"],
        lora_alpha=cfg["training"]["lora_alpha"],
        target_modules=cfg["training"]["target_modules"],
        lora_dropout=cfg["training"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # 若环境仅支持 FP16，则强制将可训练 LoRA 参数提升至 FP32 避免 GradScaler unscale 异常
    if not use_bf16:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)})

    max_len = cfg["training"]["max_seq_length"]
    train_tokens = [tokenize_with_budget(ex, tokenizer, max_len) for ex in raw_dataset["train"]]
    val_tokens = [tokenize_with_budget(ex, tokenizer, max_len) for ex in raw_dataset["validation"]]

    train_trunc_count = sum(1 for item in train_tokens if item["was_truncated"])
    val_trunc_count = sum(1 for item in val_tokens if item["was_truncated"])

    from datasets import Dataset
    train_ds = Dataset.from_list([{k: v for k, v in item.items() if k != "was_truncated"} for item in train_tokens])
    val_ds = Dataset.from_list([{k: v for k, v in item.items() if k != "was_truncated"} for item in val_tokens])

    train_kwargs = {
        "output_dir": str(out_dir / "checkpoints"),
        "per_device_train_batch_size": cfg["training"]["batch_size"],
        "per_device_eval_batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "learning_rate": cfg["training"]["learning_rate"],
        "num_train_epochs": cfg["training"]["epochs"],
        "warmup_ratio": cfg["training"]["warmup_ratio"],
        "logging_steps": 1,
        "save_strategy": "epoch",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": "adamw_torch",
        "report_to": "none",
        "seed": cfg["training"]["seed"],
    }
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **train_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **train_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    t0 = time.time()
    train_res = trainer.train()
    duration = time.time() - t0

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    vram_peak = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

    report = {
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "precision": "bfloat16" if use_bf16 else "float16",
        "cuda_device": torch.cuda.get_device_name(0),
        "vram_peak_allocated_gb": vram_peak,
        "training_duration_s": round(duration, 2),
        "final_train_loss": float(train_res.training_loss),
        "truncation_audit": {
            "max_sequence_length_budget": max_len,
            "train_samples_total": len(train_tokens),
            "train_samples_truncated": train_trunc_count,
            "val_samples_total": len(val_tokens),
            "val_samples_truncated": val_trunc_count
        },
        "adapter_path": str(out_dir)
    }

    with open(REPORTS_DIR / "local_ai_training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 训练成功！LoRA 适配器权重已落盘至: {out_dir}\n")


if __name__ == "__main__":
    main()
