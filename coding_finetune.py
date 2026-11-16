import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Initializing fine-tuning pipeline for Code & Brainstorming on: {device}")

# 1. Load Tokenizer & Base Model safely on CPU first, then push to MPS
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cpu",
)
model = model.to(device)

# 2. Configure LoRA for Full Transformer Adaptability
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Load and Tokenize Dataset
print("Loading and tokenizing training dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")

def tokenize_function(examples):
    # Combine instruction and output for causal language modeling
    prompts = [
        f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    tokenized = tokenizer(prompts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 4. Training Arguments tuned for Apple Silicon memory limits
training_args = TrainingArguments(
    output_dir="./qwen-code-brainstorm-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=1,
    max_steps=30,
    save_strategy="steps",
    save_steps=15,
    fp16=True,
    optim="adamw_torch",
    report_to="none"
)

# 5. Initialize Standard Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Starting LoRA training execution...")
trainer.train()

# 6. Save the resulting fine-tuned adapters
adapter_output_dir = "./qwen-code-brainstorm-adapter"
print(f"Saving fine-tuned LoRA adapters to {adapter_output_dir}...")
trainer.model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
print("Fine-tuning complete! Adapters saved successfully.")