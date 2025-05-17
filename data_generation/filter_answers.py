# -*- coding: utf-8 -*-
import json
import re

with open("filtered_generated_gsm_train.json", "r") as f1:
    filt_data = json.load(f1)
with open("gsm_train_ex.json", "r") as f2:
    data = json.load(f2)
assert len(filt_data) == len(data)

hit = 0
true_data = []
for fd, da in zip(filt_data, data):
    gt_eq = fd["eqs"][-1]
    gt = float(gt_eq.split("=")[-1])
    try:
        pr = da["answer"].split("####")
        assert len(pr) == 2
        num = pr[-1].replace("</ans>", "").strip()
        assert num.find(", ") == -1
        num = num.replace(",", "")
        num = re.search(r"[-+]?\d*\.?\d+", num).group()
        pr = float(num)
        if abs(gt - pr) < 1e-3:
            hit += 1
            true_data.append(da)
    except:
        print(gt)
        print(da["answer"])
        print("-"*20)

print(hit, len(filt_data), len(true_data))
with open("gsm_train_ex_verified.json", "w") as f3:
    json.dump(true_data, f3)