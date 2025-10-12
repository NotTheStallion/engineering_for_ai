from transformers import AutoModelForCausalLM, AutoTokenizer


checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cuda"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)


messages = [{"role": "user", "content": "Who are you ?"}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)


inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate the response
outputs = model.generate(inputs, max_new_tokens=5000, temperature=0.2, top_p=0.9, do_sample=True)
response = tokenizer.decode(outputs[0])

print("Response:")
print(response)
