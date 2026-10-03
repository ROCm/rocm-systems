# Vendored D3DKMT (WDDM kernel-mode thunk) headers

These headers describe the user-mode D3DKMT thunk ABI exported by `gdi32.dll`
(`D3DKMTEnumAdapters3`, `D3DKMTQueryAdapterInfo`, `D3DKMTCloseAdapter`, ...).
They are vendored so the native-Windows agent enumerator can be built against a
plain Windows SDK without requiring the Windows Driver Kit (WDK) to be
installed.

| File          | Purpose                                                        |
| ------------- | ------------------------------------------------------------- |
| `d3dkmthk.h`  | D3DKMT thunk interfaces (the functions/structs we call)       |
| `d3dkmdt.h`   | D3DKMT data types referenced by `d3dkmthk.h`                   |
| `d3dukmdt.h`  | Shared user/kernel D3D data types referenced by `d3dkmdt.h`   |

## Provenance

Copied verbatim from the Microsoft `d3dkmthk.h` family (MIT-licensed; see the
copyright banner at the top of each file). The same copies are used by the
`rocdbgapi` `third_party/libdxg` vendoring in the Windows SDK POC under
`windows-sdk-poc/rocprofiler-register-poc/projects/rocdbgapi/third_party/libdxg/include/`.

## Dependency note

`d3dukmdt.h` includes `<winapifamily.h>` and the headers reference base Windows
types (`UINT`, `WCHAR`, `LUID`, `NTSTATUS`, `WINAPI_FAMILY_PARTITION`,
`NTDDI_VERSION`). These come from `<windows.h>`/`<winternl.h>` shipped with the
regular Windows SDK. The translation units that consume these headers therefore
include `<windows.h>` before pulling these in. Only the D3DKMT thunk subset is
needed; the WDK proper is not required.

Do not hand-edit these files - re-vendor from the upstream Microsoft headers if
an update is required so the calling-convention (`APIENTRY`) typedefs stay
correct.
