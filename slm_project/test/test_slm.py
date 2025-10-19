import sys
import os

print("current directory:", os.path.abspath(os.path.dirname(__file__)))
# os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import slm


# def test_slm_inference_capital_of_france():
#     question = "What is the capital of France?"
#     response = slm.slm_inference(question, max_new_tokens=50, temperature=0.5, top_p=0.9, device="cuda")
#     assert isinstance(response, str)
#     assert len(response) > 0
#     print(f"SLM Inference Response for '{question}': {response}")


# def test_slm_inference_to_kill_a_mockingbird():
#     question = "Who wrote 'To Kill a Mockingbird'?"
#     response = slm.slm_inference(question, max_new_tokens=75, temperature=0.8, top_p=0.85, device="cuda")
#     assert isinstance(response, str)
#     assert len(response) > 0
#     print(f"SLM Inference Response for '{question}': {response}")


# def test_slm_inference_speed_of_light():
#     question = "What is the speed of light in vacuum?"
#     response = slm.slm_inference(question, max_new_tokens=100, temperature=0.7, top_p=0.95, device="cuda")
#     assert isinstance(response, str)
#     assert len(response) > 0
#     print(f"SLM Inference Response for '{question}': {response}")


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