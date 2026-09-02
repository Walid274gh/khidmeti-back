#!/usr/bin/env python3
"""Push Khidmeti NLU v6 training kernel that loads from Kaggle dataset.
Usage: build_push_v6.py push | status | log
"""
import json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
USER, KEY = ENV["KAGGLE_USERNAME"], ENV["KAGGLE_KEY"]
HF_TOKEN = ENV["HF_TOKEN"]
SLUG = "khidmeti-nlu-train-v6"
DATASET_SLUG = f"{USER}/khidmeti-nlu-v6-data"
API = "https://www.kaggle.com/api/v1"

def call(path, body=None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:2000]}", file=sys.stderr)
        sys.exit(1)

def create_kernel_script():
    """Create kernel that loads from dataset instead of inline."""
    return '''# Khidmeti NLU v6 — loads from Kaggle dataset (70k rows)
import csv, json, os, random, subprocess, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 20260810
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "''' + HF_TOKEN + '''"
REPO = "Walidrbh27/khidmeti-nlu"
BASE = "alger-ia/dziribert"
MAX_LEN, BATCH, EPOCHS, LR = 64, 64, 6, 2e-5

# Load from Kaggle dataset
DATA_DIR = "/kaggle/input/khidmeti-nlu-v6-data"
LABELS = json.load(open(f"{DATA_DIR}/labels.json"))
I2ID = {v: i for i, v in enumerate(LABELS["intents"])}
P2ID = {v: i for i, v in enumerate(LABELS["professions"])}

def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    return ([r["text"] for r in rows],
            [I2ID[r["intent"]] for r in rows],
            [P2ID[r["profession"]] for r in rows])

tr_txt, tr_int, tr_prof = load_csv(f"{DATA_DIR}/synth_v6_final.csv")
ev_txt, ev_int, ev_prof = load_csv(f"{DATA_DIR}/eval_heldout.csv")
print(f"train={len(tr_txt)} eval={len(ev_txt)}")

tok = AutoTokenizer.from_pretrained(BASE)
def enc(texts):
    e = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

tr_ids, tr_mask = enc(tr_txt)
ev_ids, ev_mask = enc(ev_txt)

# 5% val split
idx = list(range(len(tr_txt))); random.shuffle(idx)
n_val = max(1, len(idx) // 20)
vi, ti = idx[:n_val], idx[n_val:]
def sub(t, ix): return t[torch.tensor(ix)]
val_set = (sub(tr_ids, vi), sub(tr_mask, vi), torch.tensor([tr_prof[i] for i in vi]))
train_dl = DataLoader(
    TensorDataset(sub(tr_ids, ti), sub(tr_mask, ti),
                  torch.tensor([tr_int[i] for i in ti]), torch.tensor([tr_prof[i] for i in ti])),
    batch_size=BATCH, shuffle=True)

class NLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BASE)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.intent = nn.Linear(h, len(I2ID))
        self.prof = nn.Linear(h, len(P2ID))
    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        cls = self.drop((h * m).sum(1) / m.sum(1).clamp(min=1))
        return self.intent(cls), self.prof(cls)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = NLU().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = len(train_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
ce = nn.CrossEntropyLoss()

def predict(ids, mask, bs=64):
    model.eval()
    pi, pp = [], []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            li, lp = model(ids[i:i+bs].to(dev), mask[i:i+bs].to(dev))
            pi.append(li.argmax(-1).cpu()); pp.append(lp.argmax(-1).cpu())
    return torch.cat(pi), torch.cat(pp)

best_acc, best_state = -1.0, None
for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for ids, mask, yi, yp in train_dl:
        ids, mask, yi, yp = ids.to(dev), mask.to(dev), yi.to(dev), yp.to(dev)
        li, lp = model(ids, mask)
        loss = ce(li, yi) + ce(lp, yp)
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        tot += loss.item()
    _, vp = predict(val_set[0], val_set[1])
    acc = (vp == val_set[2]).float().mean().item()
    print(f"epoch {ep+1} loss={tot/len(train_dl):.4f} val_prof_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)

# Eval heldout
ei, ep_ = predict(ev_ids, ev_mask)
yi, yp = torch.tensor(ev_int), torch.tensor(ev_prof)
int_acc_fp32 = (ei == yi).float().mean().item()
prof_acc_fp32 = (ep_ == yp).float().mean().item()
print(f"EVAL_INTENT_ACC_FP32={int_acc_fp32:.4f}")
print(f"EVAL_PROF_ACC_FP32={prof_acc_fp32:.4f}")

# ONNX int8 export
model.cpu().eval()
os.makedirs("hf_out", exist_ok=True)
d = torch.zeros(2, MAX_LEN, dtype=torch.long)
torch.onnx.export(
    model, (d, d.clone().fill_(1)), "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["intent_logits", "profession_logits"],
    dynamic_axes={n: {0: "batch", 1: "seq"} for n in ["input_ids", "attention_mask"]},
    opset_version=17, dynamo=False)
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic("model.onnx", "hf_out/model.int8.onnx", weight_type=QuantType.QInt8)
os.remove("model.onnx")

# int8 eval (gate)
import onnxruntime as ort
sess = ort.InferenceSession("hf_out/model.int8.onnx", providers=["CPUExecutionProvider"])
qi, qp = [], []
ids_np, mask_np = ev_ids.numpy().astype(np.int64), ev_mask.numpy().astype(np.int64)
for i in range(0, len(ids_np), 32):
    li, lp = sess.run(None, {"input_ids": ids_np[i:i+32], "attention_mask": mask_np[i:i+32]})
    qi.append(li.argmax(-1)); qp.append(lp.argmax(-1))
qi, qp = np.concatenate(qi), np.concatenate(qp)
int_acc = float((qi == yi.numpy()).mean())
prof_acc = float((qp == yp.numpy()).mean())
print(f"EVAL_INTENT_ACC_INT8={int_acc:.4f}")
print(f"EVAL_PROF_ACC_INT8={prof_acc:.4f}")
print("GATE_90_PASS=" + ("YES" if prof_acc >= 0.90 else "NO"))

profs = LABELS["professions"]
per = {p: [0, 0] for p in profs}
for g, p in zip(yp.numpy(), qp):
    per[profs[g]][1] += 1
    per[profs[g]][0] += int(g == p)
for p, (c, n) in per.items():
    if n: print(f"  prof {p}: {c}/{n}")
for t, g, p in [(ev_txt[k], profs[yp[k]], profs[qp[k]]) for k in range(len(ev_txt)) if qp[k] != yp[k].item()]:
    print(f"  MISS [{g}->{p}] {t}")

# Gate + upload
if prof_acc < 0.90:
    print("GATE_FAILED_NO_UPLOAD — poids NON poussés (gate v6 = 90%)")
    sys.exit(0)

tok.save_pretrained("hf_out")
json.dump(LABELS, open("hf_out/labels.json", "w"), ensure_ascii=False, indent=2)
json.dump({"base_model": BASE, "max_len": MAX_LEN, "opset": 17, "trained": "2026-08-10",
           "version": "v6", "data_size": "70k rows (10.7× v5)",
           "train_rows": len(tr_txt), "eval_rows": len(ev_txt), "seed": SEED,
           "intent_acc_int8": int_acc, "prof_acc_int8": prof_acc,
           "gate": "prof_acc >= 0.90",
           "augmentations": ["regional_dialects", "arabizi", "synonyms", "typos", "context_modifiers"],
           "inputs": ["input_ids", "attention_mask"],
           "outputs": ["intent_logits", "profession_logits"]},
          open("hf_out/meta.json", "w"), indent=2)
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="hf_out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
# Auto-restart Space to pick up new weights
try:
    from huggingface_hub import HfApi as _HfApi2
    _HfApi2(token=HF_TOKEN).restart_space(repo_id="Walidrbh27/khidmeti-ai", repo_type="space")
    print("SPACE_RESTARTED=Walidrbh27/khidmeti-ai")
except Exception as _e:
    print(f"SPACE_RESTART_SKIP={_e}")

'''

def push():
    src = create_kernel_script()
    print(f"Kernel script size: {len(src)} bytes ({len(src)/1024:.1f} KB)")

    payload = {
        "slug": f"{USER}/{SLUG}",
        "newTitle": "Khidmeti NLU v6 Training",
        "text": src,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",
        "datasetDataSources": [DATASET_SLUG],
        "competitionDataSources": [],
        "kernelDataSources": [],
    }

    result = call("/kernels/push", payload)
    print(json.dumps(result, indent=2))
    print(f"\nKernel pushed! Monitor at:")
    print(f"  https://www.kaggle.com/code/{USER}/{SLUG}")

def status():
    print(json.dumps(call(f"/kernels/status?userName={USER}&kernelSlug={SLUG}")))

def log():
    out = call(f"/kernels/output?userName={USER}&kernelSlug={SLUG}")
    raw = out.get("log") or "[]"
    entries = json.loads(raw) if isinstance(raw, str) else raw
    for e in entries:
        sys.stdout.write(e.get("data", ""))

if len(sys.argv) < 2:
    print("Usage: python3 build_push_v6.py push | status | log")
    sys.exit(1)

{"push": push, "status": status, "log": log}[sys.argv[1]]()
