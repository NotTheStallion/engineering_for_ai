from slm import slm_inference, DEVICE, MODEL_NAME, MODEL, TOKENIZER
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from typing import Union



app = FastAPI()

# @note: by adding @async before def, the function becomes asynchronous and doesn't block other requests while it's running

@app.get("/") # Whenever a user uses the root endpoint, fastAPI calls this function
def read_root():
    return {"Model": "OK", "Tokenizer": "OK", "Device": DEVICE, "Model Name": MODEL_NAME}


# @app.get("/items/{item_id}") # Whenever a user uses the /items/{item_id} endpoint, fastAPI calls this function (path parameter)

@app.get("/inference/") # Whenever a user uses the /inference endpoint, fastAPI calls this function (query parameter)
def ask(question: str, max_new_tokens: Union[int, None] = 500, temperature: Union[float, None] = 0.2, top_p: Union[float, None] = 0.9):
    response = slm_inference(question, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, device=DEVICE)
    # response = "This is a test response."
    return {"response": response}



if __name__ == "__main__":
    print("ok")