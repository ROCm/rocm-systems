# Runtime intercept tables

Although most tools will want to leverage the callback or buffer tracing services for tracing the HIP, HSA, and ROCTx
APIs, rocprofiler-sdk does provide access to the raw API dispatch tables. Each of the aforementioned APIs are
designed similar to the following sample.

## Dispatch Table Overview

### Forward Declaration of public C API function

```cpp
extern "C"
{
// forward declaration of public C API function
int
foo(int) __attribute__((visibility("default")));
}
```

### Internal Implementation of API function

```cpp
namespace impl
{
int
foo(int val)
{
    // real implementation
    return (2 * val);
}
}
```

### Dispatch Table Implementation

```cpp
namespace impl
{
struct dispatch_table
{
    int (*foo_fn)(int) = nullptr;
};

// invoked once: populates the dispatch_table with function pointers to implementation
dispatch_table*&
construct_dispatch_table()
{
    static dispatch_table* tbl = new dispatch_table{};
    tbl->foo_fn                = impl::foo;

    // in between above and below, rocprofiler-sdk gets passed the pointer
    // to the dispatch table and has the opportunity to wrap the function
    // pointers for interception

    return tbl;
}

// constructs dispatch table and stores it in static variable
dispatch_table*
get_dispatch_table()
{
    static dispatch_table*& tbl = construct_dispatch_table();
    return tbl;
}
}  // namespace impl
```

### Implementation of public C API function

```cpp
extern "C"
{
// implementation of public C API function
int
foo(int val)
{
    return impl::get_dispatch_table()->foo_fn(val);
}
}
```

### Dispatch Table Chaining

rocprofiler-sdk is given an opportunity within `impl::construct_dispatch_table()` to
save the original value(s) of the function pointers such as `foo_fn` and install
it's own function pointers in its place -- this results in the public C API function `foo`
calling into the rocprofiler-sdk function pointer, which then in turn, calls the original
function pointer to `impl::foo` (this is called "chaining"). Once rocprofiler-sdk
has made any necessary modifications to the dispatch table, tools which indicated
they also want access to the raw dispatch table via `rocprofiler_at_intercept_table_registration`
will be passed the pointer to the dispatch table.

## Sample

For a demo of dispatch table chaining, please see the `samples/intercept_table` example in the
[rocprofiler-sdk GitHub repository](https://github.com/ROCm/rocproifler-sdk).
