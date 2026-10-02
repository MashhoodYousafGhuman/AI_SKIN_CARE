from transformers import AutoTokenizer, AutoModelForCausalLM

print("Model load ho raha hai... (pehli baar 1-2 minute lag sakte hain)")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("✅ Model ready!\n")

while True:
    user_input = input("Write your query here: ")
    if user_input.lower() in ["exit", "quit", "band"]:
        print("Bye!")
        break

    # Prompt format
    messages = [
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Sirf naya text nikaalo (prompt remove karo)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Model: {response}\n")