# librocdxg
librocdxg is a user-mode library enabling ROCm functionality on WSL. It allows users to run GPU-accelerated Linux workloads under WSL, supporting AI, HPC, and other experimental use cases.

## Build Instructions
This project uses CMake for building:

```bash
# download windows sdk to your windows host and install (sdk can be get from https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/))
# windows-sdk might install to C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
# in wsl, the path is win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
make install
sudo ldconfig
```

## Requirement
- Latest ROCm installed
- Windows 11, version 24H1 or later
- Supported graphic cards:
  - AMD Radeon PRO W7700
  - AMD Radeon PRO W7800
  - AMD Radeon PRO W7900
  - Radeon RX 7900
  - AMD Radeon RX 9060
  - AMD Radeon RX 9070
  - AMD Radeon AI PRO R9700

## License
see the LICENSE.md file for details.

---

For questions or suggestions, please contact the maintainer or submit an issue.
