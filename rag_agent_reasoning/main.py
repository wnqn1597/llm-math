import argparse
import copy
import json
import os
import random
import time

import tqdm
import faiss
import uuid
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gms import get_reasoning_result
from manual_shots import manual_shots

def save_json(fname, obj):
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def read_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

def load_train_test_set():
    dev_data = read_json("data/gsm_test.json")
    return dev_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--temperature", type=float, default=0.0) # 0.4
    parser.add_argument("--num_samples", type=int, default=1) # 3
    parser.add_argument("--max_tokens", type=int, default=320)
    parser.add_argument("--embedding_path", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_base", type=str, required=True)
    args = parser.parse_args()
    return args

SEP = "\n"*6
STOP_TOKEN = SEP
shot_prompt_pattern = "Q: {}\n\n# solution in Python:\n\n\n{}"
ques_prompt_pattern = "Q: {}\n\n# solution in Python:\n\n\n"

class VectorStore:
    def __init__(self, args):
        embeddings_path = args.embedding_path
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.id2qa = {}

    def add_qa_list(self, qa_list, key="question"):
        if len(qa_list) == 0:
            return
        uuids = [str(uuid.uuid4()) for _ in range(len(qa_list))]
        for uid, qa in zip(uuids, qa_list):
            self.id2qa[uid] = qa
        documents = []
        for qa in qa_list:
            if key in ["question", "answer"]:
                documents.append(Document(page_content=qa[key]))
            else:
                documents.append(Document(page_content=qa["question"] + "\n"*2 + qa["answer"]))
        self.vector_store.add_documents(documents=documents, ids=uuids)

    def retrieve(self, query_str, topk=4):
        results = self.vector_store.similarity_search(query_str, k=topk)
        ret = []
        for res in results:
            qa = self.id2qa[res.id]
            new_qa = copy.deepcopy(qa)
            ret.append(new_qa)
        return ret

def few_shots_prompt(test_ex, shots):
    shots_prompts = [shot_prompt_pattern.format(shot["question"], shot["answer"]) for shot in shots]
    test_prompt = ques_prompt_pattern.format(test_ex["question"])
    prompts = shots_prompts + [test_prompt]
    return SEP.join(prompts)

def get_completion(prompts, args):
    response = client.completions.create(
        model=args.model,
        prompt=prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        stop=STOP_TOKEN,
    )
    return response

def batch_query_engine(prompts, args):
    resps = get_completion(prompts, args)
    resps = resps.choices
    resps = [item.text for item in resps]
    resps = [
        resps[(i * args.num_samples):((i + 1) * args.num_samples)]
        for i in range(len(prompts))
    ]
    assert len(resps) == len(prompts)
    return resps

def main(args):
    data = load_train_test_set()
    num_samples = len(data)
    vector_store = VectorStore(args)
    vector_store.add_qa_list(manual_shots, "all")

    batch_sz = 10
    num_batch = (num_samples // batch_sz) + 1
    hit = 0
    local_accs = []
    cache = []
    for bidx in tqdm.tqdm(range(num_batch)):
        st, ed = bidx * batch_sz, (bidx+1) * batch_sz
        sub_data = data[st:ed]
        prompts = []
        for sd in sub_data:
            shots = manual_shots
            prompts.append(few_shots_prompt(sd, shots))
        response = batch_query_engine(prompts, args)
        prompts = []
        for sd, res_list in zip(sub_data, response):
            query = sd["question"] + "\n"*2 + res_list[0]
            shots = vector_store.retrieve(query, 8)
            prompts.append(few_shots_prompt(sd, shots))
        response = batch_query_engine(prompts, args)

        hit_qa = []
        local_hit = 0
        for sd, res_list in zip(sub_data, response):
            assert isinstance(res_list, list)
            preds = [get_reasoning_result(res) for res in res_list]
            gt = float(sd["label"].replace(",", "").strip())
            example_hit = False
            hit_answers = []
            for pred, res in zip(preds, res_list):
                if isinstance(pred, float) and abs(pred - gt) < 1e-3:
                    example_hit = True
                    hit_answers.append(res)
            if example_hit:
                hit += 1
                local_hit += 1
                hit_qa.append({"question": sd["question"], "answer": random.choice(hit_answers)})
            else:
                cache_item = {
                    "preds": preds,
                    "label": gt,
                    "answers": res_list,
                }
                cache.append(cache_item)
        local_acc = local_hit/len(sub_data)*100
        print(f"local acc: {local_acc:.2f}%")
        local_accs.append(local_acc)
        vector_store.add_qa_list(hit_qa, "all")
        # vector_store.add_qa_list(hit_qa)
        # time.sleep(0.1)

    print(f"GLOBAL ACC: {hit/num_samples*100:.2f}%.")
    print(local_accs)

    save_json("wrong.json", cache)


if __name__ == '__main__':
    args = get_args()
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        max_retries=3,
    )
    main(args)
