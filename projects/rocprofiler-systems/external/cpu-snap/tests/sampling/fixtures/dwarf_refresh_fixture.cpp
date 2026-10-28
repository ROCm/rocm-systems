// Shared library loaded by test_dwarf_unwinder.cpp after the unwinder exists:
// a frame inside it is unknown to libdwfl until the module list is refreshed.
extern "C" {

__attribute__((noinline)) void
fixture_call(void (*callback)(void*), void* arg)
{
    callback(arg);
    asm volatile("" ::: "memory");  // keep this frame: no tail call
}

}
