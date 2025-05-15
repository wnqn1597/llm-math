# -*- coding: utf-8 -*-
import time
from transformers import Qwen2Tokenizer
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://114.212.85.164:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
tokenizer = Qwen2Tokenizer.from_pretrained("../test/qwq-32b")
eqs = ['8*18=144', '144*3=432']

def get_gen_eqs(EQS):
    instruction = "Use the following equations to generate a math problem.\n"
    prompts = []
    for eqs in EQS:
        user_content = instruction + " ".join(eqs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    start = time.time()
    response = client.completions.create(
        model="QwQ-32B",
        prompt=prompts,
        max_tokens=3600,
        temperature=0.6,
        top_p=0.95,
    )
    print(f"Elapsed: {time.time() - start:.2f} s.")
    texts = [ch.text for ch in response.choices]
    return texts





