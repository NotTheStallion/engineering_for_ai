import os
os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from src import slm

def test_slm_inference():
    question = "What is the capital of France?"
    response = slm.slm_inference(question, max_new_tokens=50, temperature=0.2, top_p=0.9, device="cpu")
    assert isinstance(response, str)
    assert len(response) > 0
    print("SLM Inference Response:", response)

def test_tokenize_decode():
    text = "Hello, world!"
    tokens = slm.tokenize(text, device="cpu")
    assert isinstance(tokens, slm.torch.Tensor)
    decoded_text = slm.decode(tokens[0])
    assert isinstance(decoded_text, str)
    assert "Hello" in decoded_text
    print("Tokenize and Decode Test Passed")

def test_strip():
    text = "<|im_start|>assistant This is a test response. <|im_end|>"
    stripped_text = slm.strip(text)
    assert stripped_text == "This is a test response."
    print("Strip Function Test Passed")