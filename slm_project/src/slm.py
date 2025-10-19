from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Literal


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cuda"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

def tokenize(text: str, device: Literal["cuda", "cpu"] = "cuda") -> torch.Tensor:
    return TOKENIZER.encode(text, return_tensors="pt").to(device)

def decode(tokens: torch.Tensor)->str:
    return TOKENIZER.decode(tokens)

def strip(text: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

def slm_inference(message: str, max_new_tokens: int = 500, temperature: float = 0.2, top_p: float = 0.9, device: Literal["cuda", "cpu"] = "cuda") -> str:
    system_prompt = "You are a helpful AI assistant made by The Watcher. Answer as concisely as possible. Your name is SmolWatcher."
    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{message}"}]
    input_text = TOKENIZER.apply_chat_template(prompt, tokenize=False)
    inputs = tokenize(input_text, device=device)
    outputs = MODEL.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True)
    response = decode(outputs[0])
    response = strip(response)
    return response


def fix_function():
    pass
