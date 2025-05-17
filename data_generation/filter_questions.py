# -*- coding: utf-8 -*-
import json

with open("generated_gsm_train.json", "r") as f:
    data = json.load(f)
abnormal = 0
not_finish_cnt = 0
tbd = 0
hit = 0
multi = 0
naive = 0
filt_data = []
for da in data:
    txt = da["texts"].strip()
    if not txt.startswith("**"):
        abnormal += 1
        continue
    if txt.startswith("**Math Problem:**") or txt.startswith("**Problem:**"):
        splits = txt.split("\n")
        if len(splits) < 3:
            not_finish_cnt += 1
        elif splits[2] != "":
            if splits[1].startswith("Start"):
                naive += 1
                continue
            sol_idx = -1
            for i, sp in enumerate(splits):
                if sp.startswith("**Solution") or sp.startswith("**Step-by-") or sp.startswith("**Answer"):
                    sol_idx = i
                    break
            if sol_idx != -1:
                qu = "".join(splits[1:sol_idx])
                if qu.count("?") > 1:
                    multi += 1
                    continue
                filt_data.append(da)
            else:
                not_finish_cnt += 1
        else:
            hit += 1
            filt_data.append(da)
    else:
        not_finish_cnt += 1
    # print(txt)

print(abnormal)
print(not_finish_cnt)
print(multi)
print(naive)
print(len(filt_data))

for da in filt_data:
    da["texts"] = da["texts"].replace("\n", "").replace("**", "")
with open("filtered_generated_gsm_train.json", "w") as f:
    json.dump(filt_data, f)