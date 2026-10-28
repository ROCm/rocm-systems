// Fixture shared library for ProcMapsLldMismatch tests; built with
// `-fuse-ld=lld -Wl,-z,noseparate-code` (see tests/symbols/CMakeLists.txt)
// so its PT_LOAD segment has p_offset != p_vaddr.

extern "C" int cpu_snap_lld_fixture_exported_fn(int x)
{
    return x + 1;
}

namespace
{
int lld_fixture_static_fn(int x)
{
    return x * 2;
}
}  // namespace

extern "C" int cpu_snap_lld_fixture_call_static(int x)
{
    return lld_fixture_static_fn(x);
}
