# -*- coding: utf-8 -*-
import time
import tqdm
from transformers import Qwen2Tokenizer
from openai import OpenAI
import json
import random
import re
from collections import defaultdict
import numpy as np

eq_pat = re.compile(r"<<([0-9\+\-\*\/\.\=]+)>>")
num_pat = re.compile(r"-?\d*\.?\d+")

def permute(num):
    if num in ["0", "00"]:
        return num
    if num in ["1", "2", "3"]:
        return random.choice(["1", "2", "3", "4"])
    num_ = float(num)
    num_ += np.random.randn() * abs(num_) * 0.2
    if "." in num:  # float
        num_ = f"{num_:.2f}"
    else:
        num_ = int(num_)
        if num_ == 0:
            num_ = np.random.choice([1,1,2,3])
        num_ = str(num_)
    return num_



def gen(eqs: list[str]):
    new_eqs = []
    num_pat = re.compile(r"\d*\.?\d+")
    num_set = set()
    maps = defaultdict(list)
    src_nums = set()
    for eq in eqs:
        new_eq, tmp = eq.split("=", 1)
        if new_eq == tmp:
            continue
        nums = re.findall(num_pat, eq)
        nums_fmt = [f"{float(n):.2f}" for n in nums]
        operands, res = nums[:-1], nums[-1]
        operands_fmt, res_fmt = nums_fmt[:-1], nums_fmt[-1]

        for op, op_fmt in zip(operands, operands_fmt):
            if op_fmt not in num_set:
                src_nums.add(op_fmt)
                maps[op_fmt].append(permute(op))
            l, r = re.search(op, new_eq).span()
            new_eq = new_eq[:l] + "{}" + new_eq[r:]
        new_ops = [random.choice(maps[op_fmt]) for op_fmt in operands_fmt]
        new_eq = new_eq.format(*new_ops)
        try:
            new_res = str(eval(new_eq))
        except ZeroDivisionError:
            return []
        new_res_fmt = f"{float(new_res):.2f}"
        if "." in new_res:
            new_res = str(float(new_res_fmt))
        if new_res.endswith(".0") or new_res.endswith(".00"):
            new_res = str(int(float(new_res)))
        new_eq += "=" + new_res
        maps[res_fmt].append(new_res)
        new_eqs.append(new_eq)
        num_set = num_set.union(set(nums_fmt))
    return new_eqs


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
tokenizer = Qwen2Tokenizer.from_pretrained("qwq-32b")

def get_gen_problems(DATA):
    instruction = "Use the following equations to generate a math problem.\n"
    prompts = []
    for data in DATA:
        eqs = data["eqs"]
        user_content = instruction + " ".join(eqs)
        messages = [
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    start = time.time()
    response = client.completions.create(
        model="QwQ-32B",
        prompt=prompts,
        max_tokens=3900,
        temperature=0.6,
        top_p=0.95,
    )
    print(f"Elapsed: {time.time() - start:.2f} s.")
    texts = [ch.text for ch in response.choices]

    RET = []
    assert len(DATA) == len(texts)
    for data, text in zip(DATA, texts):
        text_sp = text.split("</think>")
        if len(text_sp) == 1:
            # no end
            continue
        data["texts"] = text_sp[-1]
        RET.append(data)
    return RET

def main():
    with open("gsm8k_train.json", "r") as f:
        data = json.load(f)
    with open("gsm8k_train_origin.json", "r") as f:
        data_origin = json.load(f)
    assert len(data) == len(data_origin)
    DATA = []
    EQS = []
    for i, da in tqdm.tqdm(enumerate(data), total=len(data)):
        origin_question = data_origin[i]["question"]
        origin_answer = data_origin[i]["answer"]
        origin_eqs = re.findall(eq_pat, origin_answer)
        if len(origin_eqs) == 0:
            continue
        if len(origin_eqs) > 5:
            num_samples = 2
        else:
            num_samples = 4
        for _ in range(num_samples):
            eqs = gen(origin_eqs)
            if len(eqs) > 0:
                obj = {
                    "origin_idx": i,
                    "origin_question": origin_question,
                    "origin_answer": origin_answer,
                    "origin_eqs": origin_eqs,
                    "eqs": eqs
                }
                EQS.append(obj)

        if len(EQS) > 50:
            RES = get_gen_problems(EQS)
            DATA.extend(RES)
            with open("generated_gsm_train.json", "w", encoding="utf-8") as f:
                json.dump(DATA, f)
            EQS = []
    DATA.extend(get_gen_problems(EQS))
    with open("generated_gsm_train.json", "w", encoding="utf-8") as f:
        json.dump(DATA, f)


if __name__ == '__main__':
    main()
