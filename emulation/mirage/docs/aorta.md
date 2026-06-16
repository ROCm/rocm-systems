- demo simple kernel
- run simple 
- run simple aorta training recipe
- run single, multinode, 
- pip install git+https://github.com/ROCm/aorta.git
- aorta 
- run single node training job
```bash
cargo install --path .
cd emulation/mirage
source .venv-mi350/bin/activate
ROCM_HOME=$(rocm-sdk path --root)

pip install git+https://github.com/ROCm/aorta.git

mirage profile create --num-nodes 1 --gpus-per-node 2 --agent MI350X double

mirage run --daemon --profile double -- torchrun --standalone --nproc_per_node=2 $(which aorta) triage run --recipe /home/arosa/rocm-systems/emulation/mirage/tests/aorta.yml
 ```

 ## simplifiy talking about emulators
 dbt = gpu aceleratoed
 rocjitsu = pure cpu


 ## why do you use an emulator?
 - acurate functional
 - future timing model
 