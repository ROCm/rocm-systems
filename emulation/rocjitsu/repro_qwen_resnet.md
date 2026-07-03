# Qwen BF16 and ResNet FP16 rocjitsu repro on sharkmi300x2

This repro uses the same `sharkmi300x2` environment as `repro.md`, but runs the
two current acceptance workloads:

- `Qwen/Qwen2.5-0.5B` in BF16.
- `torchvision` ResNet50 with FP16 model parameters and FP16 input.

Both workloads should run natively on the physical MI300X and through rocjitsu
as a gfx950 guest on the gfx942 host. Use persistent DBT caches for the guest
runs so timing reflects cached execution rather than first-run translation.

## 1. Sync the current tree

From this directory on the local machine:

```bash
branch=$(git branch --show-current | tr '/ ' '--')
remote=/tmp/rocjitsu-${branch:-detached}

rsync -a --delete \
  --exclude build \
  --exclude 'third_party/*-build' \
  --exclude 'third_party/*-subbuild' \
  --exclude .git \
  --exclude .env \
  ./ sharkmi300x2:${remote}/
```

## 2. Build with the repro ROCm environment

On `sharkmi300x2`:

```bash
ssh sharkmi300x2 -t "export remote=$remote; bash -l"

source ~/resnet_torch/.env/bin/activate
cd "$remote"

root=$(rocm-sdk path --root)
bin=$(rocm-sdk path --bin)
cmake_prefix=$(rocm-sdk path --cmake)

export ROCM_PATH="$root"
export PATH="$bin:$PATH"
export LD_LIBRARY_PATH="$root/lib:${LD_LIBRARY_PATH:-}"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$cmake_prefix" \
  -DHSA_RUNTIME64="$root/lib/libhsa-runtime64.so" \
  -DAMDCLANG="$(which amdclang++)"

cmake --build build --target rocjitsu_bin rocjitsu_kmd_shim rocjitsu_hooks -j"$(nproc)"
```

## 3. Qwen BF16 native baseline

```bash
ROCR_VISIBLE_DEVICES=0 python - <<'PY'
import time
import torch
from transformers import AutoModelForCausalLM

input_ids = torch.tensor([[9707, 11, 419, 374, 1296]], device="cuda", dtype=torch.long)

print("torch", torch.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).eval().cuda()

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    output = model(input_ids=input_ids, use_cache=False)
torch.cuda.synchronize()

logits = output.logits.detach().float().cpu().contiguous()
print("shape", tuple(logits.shape))
print("finite", bool(torch.isfinite(logits).all().item()))
print("last_argmax", int(logits[0, -1].argmax().item()))
print("elapsed_sec", round(time.time() - start, 6))
torch.save(logits, "/tmp/qwen-bf16-native-logits.pt")
PY
```

Expected native signal:

```text
device AMD Instinct MI300X
shape (1, 5, 151936)
finite True
last_argmax 13
```

## 4. Qwen BF16 rocjitsu guest run

```bash
export ROCJITSU_DBT_CACHE_DIR=/tmp/rocjitsu-qwen-full-cache-fixed
export ROCJITSU_DBT_DUMP_ALL=0

ROCR_VISIBLE_DEVICES=0 build/tools/rocjitsu/rocjitsu \
  --config configs/guest_gfx950_on_gfx942.json -- python - <<'PY'
import time
import torch
from transformers import AutoModelForCausalLM

input_ids = torch.tensor([[9707, 11, 419, 374, 1296]], device="cuda", dtype=torch.long)

print("torch", torch.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).eval().cuda()

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    output = model(input_ids=input_ids, use_cache=False)
torch.cuda.synchronize()

logits = output.logits.detach().float().cpu().contiguous()
print("shape", tuple(logits.shape))
print("finite", bool(torch.isfinite(logits).all().item()))
print("last_argmax", int(logits[0, -1].argmax().item()))
print("elapsed_sec", round(time.time() - start, 6))
torch.save(logits, "/tmp/qwen-bf16-dbt-logits.pt")
PY
```

Expected guest signal:

```text
device AMD Instinct MI350X
shape (1, 5, 151936)
finite True
last_argmax 13
```

The cached guest forward should be well under 10x the native forward time.

Optional Qwen logit comparison:

```bash
python - <<'PY'
import torch

native = torch.load("/tmp/qwen-bf16-native-logits.pt", map_location="cpu", weights_only=False)
dbt = torch.load("/tmp/qwen-bf16-dbt-logits.pt", map_location="cpu", weights_only=False)
diff = (native - dbt).abs()
last = diff[0, -1]

print("shape_equal", tuple(native.shape) == tuple(dbt.shape), tuple(native.shape))
print("logit_max", float(diff.max()))
print("logit_mean", float(diff.mean()))
print("last_logit_max", float(last.max()))
print("last_logit_mean", float(last.mean()))
print("last_argmax", int(native[0, -1].argmax()), int(dbt[0, -1].argmax()))
print("top10_native", native[0, -1].topk(10).indices.tolist())
print("top10_dbt", dbt[0, -1].topk(10).indices.tolist())
PY
```

Current known-good comparison is approximately:

```text
logit_max 0.234375
logit_mean 0.0279054716
last_logit_max 0.140625
last_logit_mean 0.0231630187
last_argmax 13 13
```

## 5. ResNet50 FP16 native baseline

```bash
ROCR_VISIBLE_DEVICES=0 python - <<'PY'
import time
import torch
from torchvision.models import ResNet50_Weights, resnet50

print("torch", torch.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))

torch.manual_seed(0)
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().half().cuda()
x = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()

y = y.detach().float().cpu().contiguous()
print("dtype", next(model.parameters()).dtype, x.dtype)
print("shape", tuple(y.shape))
print("finite", bool(torch.isfinite(y).all().item()))
print("argmax", int(y.argmax(dim=1).item()))
print("elapsed_sec", round(time.time() - start, 6))
torch.save(y, "/tmp/resnet-fp16-native-output.pt")
PY
```

Expected native signal:

```text
device AMD Instinct MI300X
dtype torch.float16 torch.float16
shape (1, 1000)
finite True
argmax 21
```

## 6. ResNet50 FP16 rocjitsu guest run

```bash
export ROCJITSU_DBT_CACHE_DIR=/tmp/rocjitsu-resnet-fp16-cache-goal
export ROCJITSU_DBT_DUMP_ALL=0

ROCR_VISIBLE_DEVICES=0 build/tools/rocjitsu/rocjitsu \
  --config configs/guest_gfx950_on_gfx942.json -- python - <<'PY'
import time
import torch
from torchvision.models import ResNet50_Weights, resnet50

print("torch", torch.__version__, "hip", torch.version.hip)
print("device", torch.cuda.get_device_name(0))

torch.manual_seed(0)
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().half().cuda()
x = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()

y = y.detach().float().cpu().contiguous()
print("dtype", next(model.parameters()).dtype, x.dtype)
print("shape", tuple(y.shape))
print("finite", bool(torch.isfinite(y).all().item()))
print("argmax", int(y.argmax(dim=1).item()))
print("elapsed_sec", round(time.time() - start, 6))
torch.save(y, "/tmp/resnet-fp16-dbt-output.pt")
PY
```

Expected guest signal:

```text
device AMD Instinct MI350X
dtype torch.float16 torch.float16
shape (1, 1000)
finite True
argmax 21
```

Optional ResNet output comparison:

```bash
python - <<'PY'
import torch

native = torch.load("/tmp/resnet-fp16-native-output.pt", map_location="cpu", weights_only=False)
dbt = torch.load("/tmp/resnet-fp16-dbt-output.pt", map_location="cpu", weights_only=False)
diff = (native - dbt).abs()

print("shape_equal", tuple(native.shape) == tuple(dbt.shape), tuple(native.shape))
print("finite", bool(torch.isfinite(native).all()), bool(torch.isfinite(dbt).all()))
print("argmax", int(native.argmax(dim=1).item()), int(dbt.argmax(dim=1).item()))
print("output_max", float(diff.max()))
print("output_mean", float(diff.mean()))
print("top5_native", native[0].topk(5).indices.tolist())
print("top5_dbt", dbt[0].topk(5).indices.tolist())
PY
```

Current known-good comparison is approximately:

```text
output_max 0.0068359375
output_mean 0.000868371979
argmax 21 21
top5_native [21, 22, 92, 127, 23]
top5_dbt [21, 22, 92, 127, 23]
```
