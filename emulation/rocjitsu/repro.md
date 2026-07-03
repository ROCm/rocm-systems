# ResNet rocjitsu repro on sharkmi300x2

These steps reproduce a torchvision ResNet50 run with the ROCm/PyTorch
environment already installed on `sharkmi300x2`.

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

## 2. Build with the ResNet ROCm environment

On `sharkmi300x2`:

```bash
# ssh to sharkmi300x2 with remote environment variable set
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

## 3. Run the ResNet50 check

Native run:

```bash
ROCR_VISIBLE_DEVICES=0 python - <<'PY'
import time
import torch
from torchvision.models import ResNet50_Weights, resnet50

print('torch', torch.__version__, 'hip', torch.version.hip)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))

torch.manual_seed(0)
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()
x = torch.randn(1, 3, 224, 224, device='cuda')
print('dtype', next(model.parameters()).dtype, x.dtype)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print('output_shape', tuple(y.shape))
print('finite', bool(torch.isfinite(y).all().item()))
print('argmax', int(y.argmax(dim=1).item()))
print('elapsed_sec', round(time.time() - start, 3))
PY
```

rocjitsu gfx950-on-gfx942 run:

```bash
ROCR_VISIBLE_DEVICES=0 ./build/tools/rocjitsu/rocjitsu \
  --config configs/guest_gfx950_on_gfx942.json -- python - <<'PY'
import time
import torch
from torchvision.models import ResNet50_Weights, resnet50

print('torch', torch.__version__, 'hip', torch.version.hip)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))

torch.manual_seed(0)
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()
x = torch.randn(1, 3, 224, 224, device='cuda')
print('dtype', next(model.parameters()).dtype, x.dtype)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print('output_shape', tuple(y.shape))
print('finite', bool(torch.isfinite(y).all().item()))
print('argmax', int(y.argmax(dim=1).item()))
print('elapsed_sec', round(time.time() - start, 3))
PY
```

Expected result: native reports `AMD Instinct MI300X`, rocjitsu reports the
guest `AMD Instinct MI350X`, and both should print:

```text
dtype torch.float32 torch.float32
output_shape (1, 1000)
finite True
argmax 21
```
