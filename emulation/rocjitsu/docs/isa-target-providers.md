# ISA Target Providers

Each final executable or shared object selects one static ISA target set. The
resulting `IsaTargetRegistry` is an immutable view over constexpr descriptors;
there is no process-global registry, runtime registration, or plugin ABI.
Separate linked images may select different subsets without creating a global
union.

Providers simplify source-integrated new-product introduction (NPI) and
per-image composition. They do not make RocJITsu open-world: a target with new
semantics may still require changes to public enums, ISA traits, code-object
handling, analysis, DBT, patching, or simulation in the downstream source tree.

## Defining a provider

A provider header publishes its complete constexpr descriptor. Its aliases,
GPU metadata, and decoder factory are all static:

```cpp
std::unique_ptr<rocjitsu::Decoder> create_target_decoder();

inline constexpr IsaTargetDescriptor kTargetDescriptor{
    .id = "vendor-next",
    .architecture_id = ROCJITSU_CODE_ARCH_VENDOR_NEXT,
    .gpu_targets = kGpuTargets,
    .decoder_factory = &create_target_decoder,
    .supports_execution = true,
};
```

The canonical string ID and aliases identify the target within the selected
registry. The optional architecture and GPU bindings connect it to RocJITsu's
closed public enums, offload-bundle processor names, and ELF machine values.

The header remains safe to include normally. An opt-in section after its
include guard exposes the descriptor to registry composition:

```cpp
#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(vendor::kTargetDescriptor)
#endif
```

The provider source owns decoder implementation details:

```cpp
std::unique_ptr<rocjitsu::Decoder> create_target_decoder() {
  return make_isa_decoder<VendorIsa>();
}
```

Mark the ISA implementation target as a provider. The provider source is
compiled with that existing target; CMake records only its declaration header.

```cmake
rj_add_object_library(vendor_isa decoder.cpp target_provider.cpp)
rj_add_isa_target_provider(
    vendor_isa
    HEADER vendor/target_provider.h
)
```

Registry construction validates the selected descriptors. A composition error
is reported by `ok()` and `error()`, and lookup fails closed.

## Composing a target set

Each final linked image defines one default registry from its exact provider
list:

```cmake
rj_add_isa_target_registry(
    errata_isa_registry
    PROVIDERS rocjitsu_isa_gfx1250_model
)
```

The simulator uses `${RJ_BUILTIN_ISA_PROVIDERS}` instead, while an
AMDGPU-specific tool can select the AMDGPU-only list. A source-integrated
downstream provider can likewise be selected by only the tool that needs it.

CMake generates only an include list of the selected provider headers. The
common composition source uses it to form a static constexpr descriptor array,
preserving the provider collection order, then constructs the registry once
over that immutable storage. Explicit descriptor factory references avoid
static constructors and `--whole-archive`. Removing a provider removes that
registry's direct dependency on its implementation.

Exactly one registry composition may be linked into an executable or shared
object because it defines that image's `default_isa_target_registry()` and
enum-based decoder entry point.

## Querying support

The default registry is the C++ source of truth for the targets present in one
linked image:

```cpp
const IsaTargetRegistry &registry = default_isa_target_registry();
if (!registry.ok())
  report_registry_error(registry.error());

for (const IsaTargetDescriptor &target : registry.targets())
  publish_supported_target(target.id, target.supports_execution);

if (registry.find(requested_target) == nullptr)
  report_unsupported_target(requested_target);
```

`targets()` enumerates descriptors in the provider order passed to
`rj_add_isa_target_registry()`. Built-in provider lists preserve the order in
which their CMake subdirectories register them. `find()` accepts a canonical
ID, alias, integrated architecture enum, or integrated GPU target.

Direct registry construction accepts only const lvalue descriptor arrays. The
array and every aliases/GPU metadata array referenced by it must remain alive
and unchanged for the registry's lifetime; provider compositions meet that
contract with `inline constexpr` storage.

The public C API does not enumerate the internal registry. It can probe the
selected subset by name with
`rj_code_decoder_create_for_target("gfx1250", &decoder)`. An unavailable target
returns a recoverable error.

## Model-only gfx1250

gfx1250 supplies separate full and model-only providers. Select exactly one in
each registry. The model-only provider links the decoder/model objects without
the execution objects, and decoded instructions have a null execution
callback. `ModelOnlyIsaTest.SymbolBoundary` and
`Gfx1250B0ToA0Library.SymbolBoundary` verify that execution and VM symbols do
not enter those final binaries.

## Adding an NPI target

1. Add the target description and provider source to the ISA implementation
   target.
2. Add public architecture or GPU enum entries only when the public enum or
   code-object APIs must address the target.
3. Select the provider in each executable or shared object that should contain
   it.
4. Follow the existing `\NPI` markers for the remaining closed-world
   integration work.

See [npi.md](npi.md) for the broader product checklist.
