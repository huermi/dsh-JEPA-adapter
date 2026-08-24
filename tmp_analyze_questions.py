"""分析 MMLU 8 学科测试题的知识点构成 (为学习资料扩容做准备)"""
import sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, REPO_ROOT)

SUBJECTS = ["global_facts", "high_school_computer_science", "elementary_mathematics",
            "us_foreign_policy", "abstract_algebra", "high_school_geography",
            "college_computer_science", "econometrics"]
PER = 12
SEED = 42

rng = np.random.RandomState(SEED)
for subj in SUBJECTS:
    t = pq.read_table(f"{REPO_ROOT}/benchmark/mmlu/{subj}/test-00000-of-00001.parquet")
    d = t.to_pydict()
    n = min(PER, len(d["question"]))
    idx = rng.choice(len(d["question"]), n, replace=False)
    print(f"\n{'='*60}\n[{subj}] ({n} 题)\n{'='*60}")
    for i in idx:
        q = d["question"][i]
        choices = d["choices"][i]
        ans = d["answer"][i]
        print(f"  Q: {q[:100]}")
        print(f"    A){choices[0][:40]}  B){choices[1][:40]}")
        print(f"    C){choices[2][:40]}  D){choices[3][:40]}  → {chr(65+ans)}")
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
