import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_scheduler)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

DEFAULT_MODEL = "gpt2"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL
    train_samples: int = 3000
    eval_samples: int = 1000
    output_dir: str = "output"
    use_lora: bool = False
    seed: int = 42


class GSM8KDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer: AutoTokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        prompt = f"Question: {item['question']}\nAnswer: "
        target = item["answer"].strip()
        full_text = f"{prompt}{target}"
        tokens = self.tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = input_ids.copy()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def ensure_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k_data(train_count: int, eval_count: int) -> Dict[str, List[Dict[str, str]]]:
    if load_dataset:
        try:
            print("Loading GSM8K dataset from Hugging Face...")
            raw = load_dataset("openai/gsm8k")
            train_data = raw["train"][:train_count]
            eval_data = raw["test"][:eval_count]
            print(f"Successfully loaded {len(train_data)} training samples and {len(eval_data)} evaluation samples.")
            return {
                "train": [{"question": x["question"], "answer": x["answer"]} for x in train_data],
                "eval": [{"question": x["question"], "answer": x["answer"]} for x in eval_data],
            }
        except Exception:
            print("Could not load GSM8K from Hugging Face. Falling back to simulated dataset.")
    else:
        print("datasets library is not installed. Using a simulated fallback dataset.")
    return simulate_gsm8k(train_count, eval_count)


def simulate_gsm8k(train_count: int, eval_count: int) -> Dict[str, List[Dict[str, str]]]:
    def make_example(i: int) -> Dict[str, str]:
        a = random.randint(1, 15)
        b = random.randint(1, 15)
        question = f"If Alice has {a} apples and buys {b} more, how many apples does she have?"
        answer = str(a + b)
        return {"question": question, "answer": answer}

    train = [make_example(i) for i in range(train_count)]
    eval_ = [make_example(i) for i in range(eval_count)]
    return {"train": train, "eval": eval_}


def build_model_and_tokenizer(model_name: str, use_lora: bool):
    print(f"Building model and tokenizer using base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    # Resize model embeddings to match tokenizer vocab size
    model.resize_token_embeddings(len(tokenizer))
    if use_lora:
        if PEFT_AVAILABLE:
            print("Applying LoRA adapter configuration for parameter-efficient fine-tuning.")
            model = prepare_model_for_int8_training(model)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        else:
            print("LoRA support is not installed. Continuing without LoRA.")
    return model, tokenizer


def evaluate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, examples: List[Dict[str, str]]) -> float:
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    correct = 0
    with torch.no_grad():
        for example in examples:
            prompt = f"Question: {example['question']}\nAnswer: "
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            outputs = model.generate(
                **encoded,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = answer.split("Answer:")[-1].strip().split("\n")[0]
            target = example["answer"].strip()
            if generated == target:
                correct += 1
    return correct / len(examples) if examples else 0.0


def train(config: TrainConfig) -> None:
    ensure_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    dataset = load_gsm8k_data(config.train_samples, config.eval_samples)
    model, tokenizer = build_model_and_tokenizer(config.model_name, config.use_lora)

    train_dataset = GSM8KDataset(dataset["train"], tokenizer)
    eval_dataset = GSM8KDataset(dataset["eval"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    print(f"Starting fine-tuning on {len(train_dataset)} train samples and {len(eval_dataset)} eval samples.")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 25 == 0 or step == len(train_loader):
                print(f"Epoch {epoch+1}/{EPOCHS}, step {step}/{len(train_loader)}, loss={running_loss/step:.4f}")

    print("Training complete. Running evaluation...")
    accuracy = evaluate(model, tokenizer, dataset["eval"])
    print(f"Evaluation exact match accuracy: {accuracy:.2%}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Model artifacts saved to {config.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K training pipeline for reasoning model prototypes.")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Operation mode.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="Pretrained base model name.")
    parser.add_argument("--train-samples", type=int, default=3000, help="Number of training samples.")
    parser.add_argument("--eval-samples", type=int, default=1000, help="Number of evaluation samples.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for saved checkpoints.")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA adapters if available.")
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model_name,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
    )

    train(config)


if __name__ == "__main__":
    main()
