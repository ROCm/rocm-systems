# nccl-device

Rust-style `no_std` wrappers over `nccl-device-sys`. The crate provides borrowed
device communicator, window, and multimem-handle types; team queries; pointer
translation; LSA barriers; and the first typed LSA reduce/copy primitive.

See the workspace `README.md` for the CUDA-Oxide and nvJitLink flow. All calls
that rely on GPU memory validity or collective participation remain `unsafe`.
