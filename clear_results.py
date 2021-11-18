import json
ALG = 'UDGCLIP'
file = f'./{ALG}_results_iid.jsonl'
results_seed0 = []
results_seed1 = []
results_seed2 = []
d = {}
with open(file, 'r') as f:
    for line in f:
        l = json.loads(line)
        seed = l['args']['seed']
        print(seed)
        if not d.get(seed):
            d[seed] = 1
        else:
            print("!!!")
print(d.keys())