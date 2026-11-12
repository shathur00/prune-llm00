import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading Model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.float16, 
    device_map=device
)

print("\nReady! Type your message (type 'exit' to quit):\n")
messages = [{"role": "user", "content": user_input}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
        )

# Slice generated response to exclude prompt tokens
generated_ids = outputs[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\nQwen > {response}\n")