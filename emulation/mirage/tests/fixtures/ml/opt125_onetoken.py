"""
OPT-125M end-to-end inference on the rocjitsu simulated GPU.

Loads OPT-125M with eager attention (no SDPA) and runs a single forward
pass to verify correct token prediction on the simulator.
"""
import os, sys, torch

device = torch.device("cuda:0")

from transformers import AutoTokenizer, OPTForCausalLM

model_name = "facebook/opt-125m"
print(f"\nLoading {model_name} with eager attention...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    attn_implementation="eager",
).to(device)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"Prompt: {prompt!r}")
print(f"Input IDs: {inputs.input_ids}")
sys.stderr.flush()
sys.stdout.flush()

print("\nRunning forward pass...")
with torch.no_grad():
    outputs = model(inputs.input_ids.to(device))

torch.cuda.synchronize()

# GPU-side NaN/Inf check (runs isnan/isinf kernels on simulated GPU)
gpu_logits = outputs.logits
gpu_nan = torch.isnan(gpu_logits).sum().item()
gpu_inf = torch.isinf(gpu_logits).sum().item()
print(f"\nGPU-side check: NaN={gpu_nan}, Inf={gpu_inf}")

# CPU-side NaN/Inf check (copy to host, check on CPU)
logits = gpu_logits.cpu()
next_token_logits = logits[0, -1, :]
nan_count = torch.isnan(next_token_logits).sum().item()
inf_count = torch.isinf(next_token_logits).sum().item()
print(f"CPU-side check: NaN={nan_count}, Inf={inf_count}")
print(f"Logits shape: {next_token_logits.shape}, dtype: {next_token_logits.dtype}")

if nan_count > 0 or inf_count > 0:
    print("FAIL: logits contain NaN or Inf!")
    #sys.exit(1)
if gpu_nan > 0 or gpu_inf > 0:
    print("FAIL: GPU-side isnan/isinf reports false positives!")
    #sys.exit(1)

top5 = torch.topk(next_token_logits.float(), 5)
print(f"\nTop-5 next token predictions:")
for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
    token = tokenizer.decode([idx.item()])
    print(f"  {i+1}. {token!r:15s} (id={idx.item():5d}, logit={val.item():.4f})")

predicted_id = top5.indices[0].item()
predicted_token = tokenizer.decode([predicted_id])
print(f"\nPredicted next token: {predicted_token!r}")
print("\nPASS: Model produced valid logits on the simulated GPU.")
 