# ConSan CDNA5 (`gfx1250`) status

Historical evidence: this ledger was not requalified during the September 2026
documentation audit. Existing results describe their recorded runs, not the
current branch. Rerun the linked procedure after relevant changes.

This ledger summarizes the latest accepted emulator evidence. Exact counts
belong to the native binaries used for those runs and must be refreshed after
relevant source, toolchain, workload, or emulator changes. Procedures and
acceptance rules are in [VALIDATION.md](VALIDATION.md).

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | SuperCollider | ConSan |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 emulator exact/complete; 846/846 accesses | 🟩 emulator exact/complete; 846/846 accesses and 80/80 barriers; required evidence; zero diagnostics |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 exact; 352/352 accesses | 🟩 exact; 352/352 accesses and 64/64 barriers |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 exact; 704/704 accesses | 🟩 exact; 704/704 accesses and 128/128 barriers |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟩 three exact rows; 920/920 accesses each | 🟩 three exact rows; 920/920 accesses and 84/84 barriers each |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 intentionally omitted from the acceptance matrix | 🩶 intentionally omitted from the acceptance matrix |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 exact; 18/18 accesses | 🟩 exact; 18/18 accesses and 8/8 barriers |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 four exact oracles; 24/24 accesses | 🟩 four exact oracles; 24/24 accesses and 8/8 barriers |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 exact; 18/18 accesses | 🟩 exact; 18/18 accesses and 8/8 barriers |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 exact; 4/4 accesses | 🟩 emulator exact/complete; 64/64 accesses, 12/12 barriers, 1/1 atomic, and 2/2 fences |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 exact; 4/4 accesses | 🟩 emulator exact/complete; 64/64 accesses, 12/12 barriers, 3/3 atomics, and 5/5 fences |
| Main E2E | P4 | hip-moi Jakub cooperative matmul (`jakub-attention`) | 🟩 four exact oracles; 70/70 accesses | 🟩 four exact oracles; 70/70 accesses and 8/8 barriers |
| Test corpus | P0 | HipKittens CDNA5 naive BF16 (`hipkittens-bf16fp32-cdna5-naive`) | 🟩 exact/complete; 26/26 accesses; qualified exact-one tile-publication miss | 🟩 exact/complete; 26/26 accesses and 4/4 barriers; qualified exact-one tile-publication miss |
| Tensile | P0 | `002_sk_mxf8gemm_explicit` (`tensile-sk-mxf8gemm-explicit`) | 🟩 exact; 70/70 accesses | 🟩 emulator exact/complete; 70/70 accesses, 32/32 barriers, 2/2 applicable communication sites, and 2/2 applicable fences; zero diagnostics |
| Tensile | P0 | `003_sk_mxf4gemm_explicit` (`tensile-sk-mxf4gemm-explicit`) | 🟩 exact; 42/42 accesses | 🟩 emulator exact/complete; 42/42 accesses, 32/32 barriers, 2/2 applicable communication sites, and 2/2 applicable fences; zero diagnostics |
| Tensile | P1 | `037_spmm_tdm_f16_transposes` (`tensile-spmm-tdm-f16-transposes`) | 🟩 exact; 672/672 accesses | 🟩 exact; 672/672 accesses and 160/160 barriers |
| Tensile | P1 | `016_spmm_tdm_all` (`tensile-spmm-tdm-all`) | 🟩 exact; 1,610/1,610 accesses | 🟩 exact; 1,610/1,610 accesses and 494/494 barriers |
| Tensile | P1 | `001_sk_mxf8f4gemm_tdm` (`tensile-sk-mxf8f4gemm-tdm`) | 🟩 18 exact rows; every shard completes 768/768 accesses | 🟩 18 exact rows; every shard completes 768/768 accesses, 204/204 barriers, 12/12 atomics, and 12/12 fences |
| Tensile | P1 | `004_sk_mxf8gemm_tdm` (`tensile-sk-mxf8gemm-tdm`) | 🟩 exact; 992/992 accesses | 🟩 36 exact rows; every shard completes 992/992 accesses, 204/204 barriers, 12/12 atomics, and 12/12 fences |
| Tensile | P1 | `007_sk_mxf4gemm_tdm` (`tensile-sk-mxf4gemm-tdm`) | 🟩 exact; 2,448/2,448 accesses | 🟩 96 exact rows; every shard completes 2,448/2,448 accesses, 544/544 barriers, 32/32 atomics, and 32/32 fences |
| Tensile | P1 | bounded Stream-K smoke (`tensile-sk-sgemm-runtime-smoke`) | 🟩 exact; 320/320 accesses | 🟩 emulator exact/complete; 320/320 accesses, 22/22 barriers, 4/4 communication sites, and 4/4 fences; zero diagnostics |
| Tensile | P2 | `000_sk_sgemm_quick` (`tensile-sk-sgemm-quick`) | 🟨 all six shards pass their first client; second clients make 8--325 exact rows but remain active at 300 s before client oracle/teardown | 🟨 all six shards pass their first client; second clients make 3--152 exact rows but remain active at 300 s before client oracle/teardown |
| Tensile | P2 | `005_sk_f8gemm_quick` (`tensile-sk-f8gemm-quick`) | 🟩 108 exact rows across all nine shapes; every shard completes 1,772/1,772 accesses | 🟩 108 exact rows across all nine shards; complete access/barrier/atomic/fence coverage; zero diagnostics |
| Tensile | P2 | `006_sk_hgemm_quick` (`tensile-sk-hgemm-quick`) | 🟩 444 exact rows across all six shapes and both asymmetric clients; complete access coverage | 🟩 444 exact rows across all six shapes and both asymmetric clients; complete access, barrier, atomic, and fence coverage |
| Tensile | P3 | `015_spmm_f8_ml` (`tensile-spmm-f8-ml`) | 🟨 six of eight exact client families on all three shards at 900 s; representative library has complete 36,760/36,760 static coverage | 🟨 exact progress of 6/8, 6/8, and 1/8 clients; timeout-only at 900 s |
| Tensile | P2 | `019_spmm_f16_sb` (`tensile-spmm-f16-sb`) | 🟨 9,546/9,546 accesses; timeout-only before first numeric row at 300 s | 🟨 9,546/9,546 accesses and 646/646 barriers; timeout-only at 300 s |
| Tensile survey | — | remaining selected configurations | 🟧 selected executable denominator no longer accepts every row | 🟧 selected executable denominator no longer accepts every row |
| PyTorch | P0 | tensor-descriptor add (`pytorch-tdm-descriptor-add`) | 🟩 exact; 29/29 accesses | 🟩 exact; 29/29 accesses and 20/20 barriers |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 exact; 28,195/28,195 accesses | 🟩 emulator exact/complete with zero diagnostics; 29,104/29,104 accesses, 8,892/8,892 barriers, and 3/3 atomics |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 emulator exact/complete; 232,476/232,476 accesses; zero diagnostics | 🟩 emulator exact/complete with zero diagnostics; 232,780/232,780 accesses and 22,846/22,846 barriers |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 exact; 48,224/48,224 accesses | 🟩 emulator exact/complete with zero diagnostics; 53,972/53,972 accesses and 12,064/12,064 barriers |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟩 BF16/FP32 exact; 23/23 accesses | 🟩 BF16/FP32 exact; 23/23 accesses |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 FP32/FP64 exact; 133/133 accesses | 🟩 FP32/FP64 exact; 175/175 accesses and 168/168 barriers |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟩 exact; 4,756/4,756 accesses | 🟩 exact; 4,756/4,756 accesses and 4,572/4,572 barriers |
| PyTorch | P1 | cluster synchronization (`pytorch-cluster-load-sync`) | 🟩 exact; 25/25 accesses | 🟩 exact; 25/25 accesses and 4/4 barriers |
| PyTorch survey | — | cluster-memory / inter-workgroup synchronization | 🟩 cluster-scope bundle accepted | 🟩 cluster-scope bundle accepted |
