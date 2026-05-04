# trainLoraPf9.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


def build_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


@dataclass
class DataCollatorForCausalLMPadManual:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("pad_token_id and eos_token_id are both None. Check tokenizer.")

        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids, batch_attention_mask, batch_labels = [], [], []

        for f in features:
            input_ids = f["input_ids"]
            attn = f["attention_mask"]
            labels = f["labels"]

            if not (len(input_ids) == len(attn) == len(labels)):
                raise ValueError(
                    f"Length mismatch: input_ids={len(input_ids)}, attention_mask={len(attn)}, labels={len(labels)}"
                )

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append(attn + [0] * pad_len)
            batch_labels.append(labels + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4, help="datasets.map num_proc")
    ap.add_argument("--dataloader_workers", type=int, default=0)
    ap.add_argument("--use_fp16", action="store_true", help="Force fp16 (V100 用 fp16 即可)")
    ap.add_argument("--no_grad_checkpointing", action="store_true", help="Disable grad checkpointing (NOT recommended)")
    ap.add_argument("--grad_checkpointing", action="store_true", help="Enable grad checkpointing (default ON)")
    
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.10)
    args = ap.parse_args()

    use_gc = True
    if args.no_grad_checkpointing:
        use_gc = False
    if args.grad_checkpointing:
        use_gc = True
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = (bf16_ok and not args.use_fp16)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if hasattr(model, "hf_device_map"):
        print("[DEBUG] hf_device_map:", model.hf_device_map)

    if use_gc:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing: ON")
    else:
        model.config.use_cache = True
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        print("[INFO] Gradient checkpointing: OFF")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    def map_fn(ex):
        messages = ex["messages"]
        output_json = ex["output_json"]
        if not isinstance(output_json, str):
            output_json = json.dumps(output_json, ensure_ascii=False)

        prompt_text = build_prompt_text(tokenizer, messages)
        eos = tokenizer.eos_token or ""
        answer_text = output_json + eos

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]

        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]

        full_len = len(prompt_ids) + len(answer_ids)
        was_trunc = 1 if full_len > args.max_length else 0

        input_ids = (prompt_ids + answer_ids)[: args.max_length]
        used_len = len(input_ids)
        attention_mask = [1] * used_len

        prompt_len = min(len(prompt_ids), used_len)
        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        input_ids = list(map(int, input_ids))
        attention_mask = list(map(int, attention_mask))
        labels = list(map(int, labels))

        if not (len(input_ids) == len(attention_mask) == len(labels)):
            raise ValueError("map_fn produced mismatched lengths.")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "__trunc__": was_trunc,
            "__full_len__": int(full_len),
            "__used_len__": int(used_len),
        }

    if args.num_workers and args.num_workers > 1:
        ds = ds.map(map_fn, remove_columns=ds.column_names, num_proc=args.num_workers)
    else:
        ds = ds.map(map_fn, remove_columns=ds.column_names)

    n = len(ds)
    if n > 0:
        n_trunc = int(sum(ds["__trunc__"]))
        full_lens = ds["__full_len__"]
        used_lens = ds["__used_len__"]

        def p50(arr):
            arr = sorted(arr)
            return arr[len(arr) // 2]

        print(f"[TRUNC] max_length={args.max_length}  trunc_rate={n_trunc/n:.3f} ({n_trunc}/{n})")
        print(f"[TRUNC] full_len: min={min(full_lens)} p50={p50(full_lens)} max={max(full_lens)}")
        print(f"[TRUNC] used_len: min={min(used_lens)} p50={p50(used_lens)} max={max(used_lens)}")

    ds = ds.remove_columns(["__trunc__", "__full_len__", "__used_len__"])

    data_collator = DataCollatorForCausalLMPadManual(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        eval_strategy="no",
        report_to="none",

        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),

        optim="adamw_torch",
        max_grad_norm=1.0,

        dataloader_num_workers=args.dataloader_workers,
        remove_unused_columns=False,

        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[OK] LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
