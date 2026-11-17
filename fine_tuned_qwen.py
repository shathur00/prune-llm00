import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./qwen-code-brainstorm-adapter"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading base model and fine-tuned adapters on {device}...")

# 1. Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cpu"
)

# 2. Layer your saved LoRA adapters on top of the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model = model.to(device)
model.eval()

print("\nFine-tuned Qwen Ready! Type your message (type 'exit' to quit):\n")
while True:
    user_input = input("You > ")
    if user_input.lower() == "exit":
        break

    # Format using chat template for Qwen Instruct models
    messages = [{"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

    # Decode only the generated response
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    print("\nQwen (Fine-tuned) >", tokenizer.decode(generated_ids, skip_special_tokens=True), "\n")