#!/usr/bin/env python3
"""Push NLU v6 training kernel with INLINE dataset (compressed efficiently)."""
import base64, gzip, json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines()
           if "=" in l and not l.strip().startswith("#"))
USER, KEY = ENV["KAGGLE_USERNAME"], ENV["KAGGLE_KEY"]
HF_TOKEN = ENV["HF_TOKEN"]
SLUG = "khidmeti-nlu-train-v6"
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

def compress_csv(path):
    """Compress CSV to base64 - sample 50% of rows to fit in 1MB."""
    import random
    lines = path.read_text().splitlines()
    header = lines[0]
    data_lines = lines[1:]

    # Sample 50% to reduce size
    random.seed(20260810)
    sampled = random.sample(data_lines, len(data_lines) // 2)

    content = header + "\n" + "\n".join(sampled) + "\n"
    compressed = gzip.compress(content.encode(), 9)
    b64 = base64.b64encode(compressed).decode()

    print(f"  {path.name}: {len(lines)} rows -> {len(sampled)+1} sampled")
    print(f"  Original: {len(content):,} bytes")
    print(f"  Compressed: {len(compressed):,} bytes ({len(compressed)/1024:.1f} KB)")
    print(f"  Base64: {len(b64):,} bytes ({len(b64)/1024:.1f} KB)")

    return b64

def create_kernel_script():
    """Create kernel with inline compressed data (50% sample)."""
    ds = ROOT / "ml" / "dataset"

    print("Compressing datasets...")
    train_b64 = compress_csv(ds / "synth_v6_final.csv")
    eval_b64 = compress_csv(ds / "eval_heldout.csv")
    labels_b64 = base64.b64encode(gzip.compress((ds / "labels.json").read_bytes(), 9)).decode()

    total_size = len(train_b64) + len(eval_b64) + len(labels_b64)
    print(f"\nTotal inline data: {total_size:,} bytes ({total_size/1024:.1f} KB)")

    if total_size > 900_000:
        print("⚠️  Still too large! Using 25% sample instead...")
        import random
        lines = (ds / "synth_v6_final.csv").read_text().splitlines()
        header = lines[0]
        data_lines = lines[1:]
        random.seed(20260810)
        sampled = random.sample(data_lines, len(data_lines) // 4)
        content = header + "\n" + "\n".join(sampled) + "\n"
        train_b64 = base64.b64encode(gzip.compress(content.encode(), 9)).decode()

        total_size = len(train_b64) + len(eval_b64) + len(labels_b64)
        print(f"  New total: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print(f"  Training on {len(sampled)+1} rows (25% sample)")

    script = f'''# Khidmeti NLU v6 - inline compressed data (sampled for size)
import base64, csv, gzip, io, json, os, random, subprocess, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 20260810
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HF_TOKEN = "{HF_TOKEN}"
REPO = "Walidrbh27/khidmeti-nlu"
BASE = "alger-ia/dziribert"
MAX_LEN, BATCH, EPOCHS, LR = 64, 64, 6, 2e-5

def unpack(b64):
    return gzip.decompress(base64.b64decode(b64)).decode()

LABELS = json.loads(unpack("{labels_b64}"))
I2ID = {{v: i for i, v in enumerate(LABELS["intents"])}}
P2ID = {{v: i for i, v in enumerate(LABELS["professions"])}}

def load_csv(data):
    rows = list(csv.DictReader(io.StringIO(unpack(data))))
    return ([r["text"] for r in rows],
            [I2ID[r["intent"]] for r in rows],
            [P2ID[r["profession"]] for r in rows])

print("Loading data...")
tr_txt, tr_int, tr_prof = load_csv("{train_b64}")
ev_txt, ev_int, ev_prof = load_csv("{eval_b64}")
print(f"train={{len(tr_txt)}} eval={{len(ev_txt)}}")

tok = AutoTokenizer.from_pretrained(BASE)
def enc(texts):
    e = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

tr_ids, tr_mask = enc(tr_txt)
ev_ids, ev_mask = enc(ev_txt)

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
    print(f"epoch {{ep+1}} loss={{tot/len(train_dl):.4f}} val_prof_acc={{acc:.4f}}")
    if acc > best_acc:
        best_acc = acc
        best_state = {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

model.load_state_dict(best_state)

ei, ep_ = predict(ev_ids, ev_mask)
yi, yp = torch.tensor(ev_int), torch.tensor(ev_prof)
print(f"EVAL_INTENT_ACC_FP32={{(ei == yi).float().mean().item():.4f}}")
print(f"EVAL_PROF_ACC_FP32={{(ep_ == yp).float().mean().item():.4f}}")

model.cpu().eval()
os.makedirs("hf_out", exist_ok=True)
d = torch.zeros(2, MAX_LEN, dtype=torch.long)
torch.onnx.export(
    model, (d, d.clone().fill_(1)), "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["intent_logits", "profession_logits"],
    dynamic_axes={{n: {{0: "batch", 1: "seq"}} for n in ["input_ids", "attention_mask"]}},
    opset_version=17, dynamo=False)
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic("model.onnx", "hf_out/model.int8.onnx", weight_type=QuantType.QInt8)
os.remove("model.onnx")

import onnxruntime as ort
sess = ort.InferenceSession("hf_out/model.int8.onnx", providers=["CPUExecutionProvider"])
qi, qp = [], []
ids_np, mask_np = ev_ids.numpy().astype(np.int64), ev_mask.numpy().astype(np.int64)
for i in range(0, len(ids_np), 32):
    li, lp = sess.run(None, {{"input_ids": ids_np[i:i+32], "attention_mask": mask_np[i:i+32]}})
    qi.append(li.argmax(-1)); qp.append(lp.argmax(-1))
qi, qp = np.concatenate(qi), np.concatenate(qp)
int_acc = float((qi == yi.numpy()).mean())
prof_acc = float((qp == yp.numpy()).mean())
print(f"EVAL_INTENT_ACC_INT8={{int_acc:.4f}}")
print(f"EVAL_PROF_ACC_INT8={{prof_acc:.4f}}")
print("GATE_90_PASS=" + ("YES" if prof_acc >= 0.90 else "NO"))

if prof_acc < 0.90:
    print("GATE_FAILED_NO_UPLOAD (sampled data, may need full dataset)")
    sys.exit(0)

tok.save_pretrained("hf_out")
json.dump(LABELS, open("hf_out/labels.json", "w"), ensure_ascii=False, indent=2)
json.dump({{"base_model": BASE, "trained": "2026-08-10", "version": "v6_sampled",
           "train_rows": len(tr_txt), "seed": SEED, "prof_acc_int8": prof_acc}},
          open("hf_out/meta.json", "w"), indent=2)
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="hf_out", repo_id=REPO)
print("HF_UPLOAD_DONE=" + REPO)
'''

    return script

def push():
    src = create_kernel_script()
    size = len(src)
    print(f"\nKernel script size: {size:,} bytes ({size/1024:.1f} KB)")

    if size > 1_000_000:
        print("❌ Script still >1MB! Cannot push.")
        sys.exit(1)

    payload = {
        "slug": f"{USER}/{SLUG}",
        "newTitle": "Khidmeti NLU v6 Training (Sampled)",
        "text": src,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",
        "datasetDataSources": [],
        "competitionDataSources": [],
        "kernelDataSources": [],
    }

    print("\nPushing to Kaggle...")
    result = call("/kernels/push", payload)
    print(json.dumps(result, indent=2))

def status():
    print(json.dumps(call(f"/kernels/status?userName={USER}&kernelSlug={SLUG}")))

def log():
    out = call(f"/kernels/output?userName={USER}&kernelSlug={SLUG}")
    raw = out.get("log") or "[]"
    entries = json.loads(raw) if isinstance(raw, str) else raw
    for e in entries:
        sys.stdout.write(e.get("data", ""))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 push_nlu_v6_inline.py push | status | log")
        sys.exit(1)

    {"push": push, "status": status, "log": log}[sys.argv[1]]()
