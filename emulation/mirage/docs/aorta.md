- demo simple kernel
- run simple 
- run simple aorta training recipe
- run single, multinode, 
- pip install git+https://github.com/ROCm/aorta.git
- aorta 
- run single node training job

```bash
alias mirage="cargo run --"
```

## setup venv

pip install

`--index-url "https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/"`

```
git+https://github.com/ROCm/aorta.git@2756fa149d5dacd63d48b05e7202e28951348c19
beautifulsoup4==4.14.3
cfgv==3.5.0
click==8.3.1
contourpy==1.3.3
coverage==7.13.4
cycler==0.12.1
distlib==0.4.0
et_xmlfile==2.0.0
execnet==2.1.2
filelock==3.25.0
fonttools==4.61.1
fsspec==2026.2.0
identify==2.6.17
iniconfig==2.3.0
Jinja2==3.1.6
kiwisolver==1.4.9
librt==0.8.1
MarkupSafe==3.0.2
matplotlib==3.10.8
mpmath==1.3.0
mypy==1.19.1
mypy_extensions==1.1.0
networkx==3.6.1
nodeenv==1.10.0
numpy==2.4.2
openpyxl==3.1.5
packaging==26.0
pandas==3.0.1
pathspec==1.0.4
pillow==12.1.1
platformdirs==4.9.2
pluggy==1.6.0
pre_commit==4.5.1
Pygments==2.19.2
pyparsing==3.3.2
pytest==9.0.2
pytest-cov==7.0.0
pytest-timeout==2.4.0
pytest-xdist==3.8.0
python-dateutil==2.9.0.post0
python-discovery==1.1.0
PyYAML==6.0.3
ruff==0.15.4
seaborn==0.13.2
setuptools==78.1.0
six==1.17.0
soupsieve==2.8.3
sympy==1.14.0
tabulate==0.9.0
torch==2.12.0.dev20260303+rocm7.2
triton-rocm==3.6.0+git9844da95
typing_extensions==4.15.0
virtualenv==21.1.0
```
## test 1
```

cd emulation/mirage

source .venv/bin/activate && cargo run -- run --profile mi350x --daemon \
  --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
  -- $(which aorta) triage run --verbose --recipe recipes/example-llm-determinism-smoke.yaml
```

## test 2
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
 