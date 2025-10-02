# Frida HIP Runtime Injection Mechanism

This document provides a detailed technical explanation of how Frida injects into the HIP runtime and traces API calls.

## Overview

Frida's ability to trace HIP API functions relies on a sophisticated injection mechanism that operates at the process level, allowing dynamic instrumentation of native libraries without modifying the target application's source code.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Target Process                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Application   │  │  Frida Agent    │  │ HIP Runtime │  │
│  │                 │  │                 │  │             │  │
│  │  main()         │  │  JavaScript     │  │ libamdhip64 │  │
│  │  ├─ hipMalloc() │  │  Engine (V8)    │  │    .so.7    │  │
│  │  ├─ hipMemcpy() │  │  ├─ Hooks       │  │             │  │
│  │  └─ hipFree()   │  │  ├─ Callbacks   │  │ hipMalloc() │  │
│  │                 │  │  └─ Logging     │  │ hipMemcpy() │  │
│  └─────────────────┘  └─────────────────┘  │ hipFree()   │  │
│                                            └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frida Core                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Process       │  │   Interceptor   │  │   Memory    │  │
│  │   Manager       │  │   Engine        │  │   Manager   │  │
│  │                 │  │                 │  │             │  │
│  │  ├─ Spawn       │  │  ├─ Hook        │  │  ├─ Read    │  │
│  │  ├─ Attach      │  │  ├─ Replace     │  │  ├─ Write   │  │
│  │  └─ Detach      │  │  └─ Unhook      │  │  └─ Alloc   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Injection Process

### 1. Process Spawning

When using `frida -f target_app`, Frida performs the following steps:

```bash
# Frida command
frida -l script.js -f /path/to/app

# Internal process
1. fork() - Create new process
2. ptrace(PTRACE_TRACEME) - Enable tracing
3. execve() - Load target application
4. PTRACE_ATTACH - Attach to child process
5. Inject agent library
```

**Technical Details:**
- Uses `ptrace()` system call for process control
- Suspends the target process before injection
- Maintains parent-child relationship for control

### 2. Agent Library Injection

Frida injects its agent library (`frida-agent.so`) into the target process:

```c
// Simplified injection process
void inject_agent(pid_t target_pid) {
    // 1. Allocate memory in target process
    void *remote_memory = allocate_remote_memory(target_pid, agent_size);

    // 2. Write agent library to remote memory
    write_remote_memory(target_pid, remote_memory, agent_data, agent_size);

    // 3. Create remote thread to execute agent
    create_remote_thread(target_pid, agent_entry_point, remote_memory);

    // 4. Wait for agent initialization
    wait_for_agent_ready(target_pid);
}
```

**Memory Layout After Injection:**
```
Target Process Memory Layout:
┌─────────────────────────────────────┐
│ 0x400000: Application Code          │
├─────────────────────────────────────┤
│ 0x7f0000000000: Heap                │
├─────────────────────────────────────┤
│ 0x7f1000000000: Frida Agent         │ ← Injected
│ ├─ JavaScript Engine (V8)           │
│ ├─ Hook Management                  │
│ ├─ Memory Access APIs               │
│ └─ Communication Layer              │
├─────────────────────────────────────┤
│ 0x7f2000000000: HIP Runtime         │
│ ├─ libamdhip64.so.7.1.70100         │
│ ├─ hipMalloc()                      │
│ ├─ hipMemcpy()                      │
│ └─ hipFree()                        │
├─────────────────────────────────────┤
│ 0x7f3000000000: Stack               │
└─────────────────────────────────────┘
```

### 3. JavaScript Engine Initialization

The injected agent initializes a JavaScript engine (V8) to execute our tracing script:

```javascript
// Frida's JavaScript runtime
const js_engine = new V8Engine();
js_engine.initialize();

// Load our tracing script
const script_source = read_file("frida-hip-tracer.js");
const script = js_engine.compile(script_source);
js_engine.execute(script);
```

**JavaScript API Bindings:**
```c
// C++ to JavaScript bindings
void register_frida_apis(JSContext *ctx) {
    // Process API
    js_register_function(ctx, "Process.getCurrentProcess", &process_get_current);
    js_register_function(ctx, "Process.getCurrentThreadId", &process_get_thread_id);

    // Module API
    js_register_function(ctx, "Process.findModuleByName", &process_find_module);
    js_register_function(ctx, "Module.enumerateExports", &module_enumerate_exports);

    // Interceptor API
    js_register_function(ctx, "Interceptor.attach", &interceptor_attach);

    // Memory API
    js_register_function(ctx, "Memory.readPointer", &memory_read_pointer);
    js_register_function(ctx, "Memory.writePointer", &memory_write_pointer);
}
```

## HIP Function Hooking

### 1. Library Detection

Our script detects the HIP runtime library:

```javascript
// Find HIP library in loaded modules
const HIP_LIBRARY = "libamdhip64.so.7.1.70100";
const module = Process.findModuleByName(HIP_LIBRARY);

// Internal Frida process:
// 1. Enumerate all loaded modules via /proc/pid/maps
// 2. Parse ELF headers to get module information
// 3. Return module object with base address and size
```

**Module Enumeration Process:**
```c
// Simplified module enumeration
typedef struct {
    char name[256];
    void *base_address;
    size_t size;
    void *entry_point;
} ModuleInfo;

ModuleInfo* enumerate_modules(pid_t pid) {
    // Read /proc/pid/maps
    FILE *maps = fopen("/proc/pid/maps", "r");

    while (fgets(line, sizeof(line), maps)) {
        // Parse memory mapping line
        // Format: start-end perms offset dev inode pathname
        // Example: 7f2000000000-7f2001000000 r-xp 00000000 08:01 123456 /opt/rocm/lib/libamdhip64.so.7.1.70100

        if (strstr(line, "libamdhip64")) {
            ModuleInfo *module = malloc(sizeof(ModuleInfo));
            sscanf(line, "%p-%p", &module->base_address, &module->size);
            strcpy(module->name, "libamdhip64.so.7.1.70100");
            // Add to module list
        }
    }

    fclose(maps);
    return module_list;
}
```

### 2. Export Table Resolution

Frida resolves function addresses from the ELF export table:

```javascript
// Get function exports from module
const exports = module.enumerateExports();
const export_func = exports.find(exp => exp.name === "hipMalloc");

// Internal Frida process:
// 1. Parse ELF header to find dynamic section
// 2. Locate symbol table (.dynsym)
// 3. Find function symbol by name
// 4. Calculate absolute address
```

**ELF Symbol Resolution:**
```c
// Simplified symbol resolution
void* resolve_symbol(void *module_base, const char *symbol_name) {
    // 1. Parse ELF header
    Elf64_Ehdr *ehdr = (Elf64_Ehdr*)module_base;

    // 2. Find dynamic section
    Elf64_Shdr *shdr = (Elf64_Shdr*)(module_base + ehdr->e_shoff);
    Elf64_Shdr *dynsym_shdr = find_section(shdr, ".dynsym");
    Elf64_Shdr *dynstr_shdr = find_section(shdr, ".dynstr");

    // 3. Search symbol table
    Elf64_Sym *symtab = (Elf64_Sym*)(module_base + dynsym_shdr->sh_offset);
    char *strtab = (char*)(module_base + dynstr_shdr->sh_offset);

    for (int i = 0; i < dynsym_shdr->sh_size / sizeof(Elf64_Sym); i++) {
        if (strcmp(strtab + symtab[i].st_name, symbol_name) == 0) {
            // 4. Calculate absolute address
            return module_base + symtab[i].st_value;
        }
    }

    return NULL;
}
```

### 3. Function Hook Installation

Frida installs hooks using code patching:

```javascript
// Install hook on function
Interceptor.attach(export_func.address, {
    onEnter: function(args) {
        // Log function entry
    },
    onLeave: function(retval) {
        // Log function exit
    }
});

// Internal Frida process:
// 1. Save original function bytes
// 2. Write jump instruction to hook handler
// 3. Set up trampoline for original function
// 4. Enable write permissions on code page
```

**Code Patching Process:**
```c
// Simplified hook installation
typedef struct {
    void *target_address;
    void *hook_handler;
    uint8_t original_bytes[16];
    void *trampoline;
} Hook;

void install_hook(void *target_addr, void *handler) {
    Hook *hook = malloc(sizeof(Hook));
    hook->target_address = target_addr;
    hook->hook_handler = handler;

    // 1. Save original bytes
    memcpy(hook->original_bytes, target_addr, 16);

    // 2. Create trampoline
    hook->trampoline = create_trampoline(target_addr, hook->original_bytes);

    // 3. Write jump instruction
    uint8_t jump_code[] = {
        0x48, 0xB8,                    // mov rax, handler
        // 8 bytes of handler address
        0xFF, 0xE0                     // jmp rax
    };

    // 4. Enable write permissions
    mprotect(target_addr, 16, PROT_READ | PROT_WRITE | PROT_EXEC);

    // 5. Write jump code
    memcpy(target_addr, jump_code, sizeof(jump_code));

    // 6. Flush instruction cache
    __builtin___clear_cache(target_addr, target_addr + 16);
}
```

### 4. Hook Handler Execution

When a hooked function is called, Frida's handler executes:

```c
// Simplified hook handler
void hook_handler(CPUContext *ctx, void *user_data) {
    // 1. Save CPU context
    CPUContext saved_context = *ctx;

    // 2. Call JavaScript onEnter callback
    js_call_callback("onEnter", ctx->rdi, ctx->rsi, ctx->rdx, ctx->rcx);

    // 3. Execute original function via trampoline
    void *result = call_trampoline(ctx);

    // 4. Call JavaScript onLeave callback
    js_call_callback("onLeave", result);

    // 5. Restore CPU context
    *ctx = saved_context;
}
```

**JavaScript Callback Execution:**
```javascript
// Our JavaScript callbacks
function onEnter(args) {
    const timestamp = getTimestamp();
    const threadId = Process.getCurrentThreadId();

    // Log function entry
    logToFile(`[${timestamp}] ENTER ${functionName}()`);
    logToCSV(timestamp, threadId, functionName, "ENTER", args, null, 0);

    // Store start time
    this.startTime = Date.now();
}

function onLeave(retval) {
    const timestamp = getTimestamp();
    const threadId = Process.getCurrentThreadId();
    const duration = Date.now() - this.startTime;

    // Log function exit
    logToFile(`[${timestamp}] LEAVE ${functionName}() [duration=${duration}ms]`);
    logToCSV(timestamp, threadId, functionName, "LEAVE", this.csvArgs, retval, duration);
}
```

## Memory Access and Safety

### 1. Safe Memory Reading

Frida provides safe memory access APIs:

```javascript
// Safe pointer dereferencing
function formatPointer(ptr) {
    if (ptr.isNull()) return "NULL";
    return "0x" + ptr.toString(16);
}

// Safe memory reading
function readString(ptr) {
    try {
        return ptr.readCString();
    } catch (e) {
        return formatPointer(ptr);
    }
}
```

**Memory Safety Implementation:**
```c
// Simplified safe memory access
bool safe_read_memory(pid_t pid, void *addr, void *buffer, size_t size) {
    // 1. Check if address is valid
    if (!is_valid_address(pid, addr, size)) {
        return false;
    }

    // 2. Use ptrace to read memory
    long result = ptrace(PTRACE_PEEKDATA, pid, addr, 0);
    if (result == -1 && errno != 0) {
        return false;
    }

    // 3. Copy data to buffer
    memcpy(buffer, &result, min(size, sizeof(long)));

    return true;
}
```

### 2. Argument Parsing

Frida parses function arguments based on calling convention:

```javascript
// Parse arguments based on x86_64 calling convention
function parseArguments(args, argTypes) {
    const parsedArgs = [];

    for (let i = 0; i < argTypes.length; i++) {
        const arg = args[i];
        const type = argTypes[i];

        try {
            if (type === "size_t") {
                parsedArgs.push(arg.toString());
            } else if (type === "int") {
                parsedArgs.push(arg.toInt32().toString());
            } else if (type === "void*") {
                parsedArgs.push(formatPointer(arg));
            } else if (type === "const char*") {
                parsedArgs.push(`"${arg.readCString()}"`);
            }
        } catch (e) {
            parsedArgs.push(formatPointer(arg));
        }
    }

    return parsedArgs;
}
```

**x86_64 Calling Convention:**
```
Register Usage:
- RDI: 1st argument
- RSI: 2nd argument
- RDX: 3rd argument
- RCX: 4th argument
- R8:  5th argument
- R9:  6th argument
- Stack: 7th+ arguments

Return Value:
- RAX: Return value
```

## Performance Considerations

### 1. Hook Overhead

Each function call incurs overhead:

```c
// Hook overhead breakdown
typedef struct {
    uint64_t context_save_time;      // ~50ns
    uint64_t js_callback_time;       // ~100-1000ns
    uint64_t trampoline_time;        // ~20ns
    uint64_t context_restore_time;   // ~50ns
} HookOverhead;

// Total overhead: ~220-1120ns per function call
```

### 2. Optimization Techniques

Frida uses several optimization techniques:

```c
// 1. Lazy hook installation
void lazy_install_hook(void *target_addr) {
    // Only install hook when function is first called
    if (!is_hook_installed(target_addr)) {
        install_hook(target_addr, hook_handler);
    }
}

// 2. Batch logging
void batch_log_entries(LogEntry *entries, size_t count) {
    // Collect multiple log entries and write them in batch
    write_log_batch(entries, count);
}

// 3. Asynchronous I/O
void async_write_log(const char *message) {
    // Write to log file asynchronously to avoid blocking
    enqueue_log_write(message);
}
```

## Security and Isolation

### 1. Process Isolation

Frida maintains process isolation:

```c
// Process isolation mechanisms
void ensure_isolation(pid_t target_pid) {
    // 1. Verify target process permissions
    if (!has_ptrace_permissions(target_pid)) {
        error("Insufficient permissions to attach to process");
    }

    // 2. Enable seccomp filtering
    enable_seccomp_filter();

    // 3. Limit system call access
    restrict_system_calls();
}
```

### 2. Memory Protection

Frida protects against memory corruption:

```c
// Memory protection mechanisms
void protect_memory(void *addr, size_t size) {
    // 1. Mark memory as read-only when not modifying
    mprotect(addr, size, PROT_READ);

    // 2. Use copy-on-write for shared memory
    enable_copy_on_write(addr, size);

    // 3. Validate memory accesses
    validate_memory_access(addr, size);
}
```

## Debugging and Troubleshooting

### 1. Hook Installation Debugging

```bash
# Enable Frida debug output
export FRIDA_DEBUG=1
frida -l script.js -f target_app

# Check hook installation
gdb -p $(pgrep target_app)
(gdb) info functions hipMalloc
(gdb) x/16i hipMalloc
```

### 2. Memory Access Debugging

```javascript
// Debug memory access
function debugMemoryAccess(ptr) {
    try {
        console.log("Address:", ptr);
        console.log("Is null:", ptr.isNull());
        console.log("Is valid:", ptr.isValid());
        console.log("Value:", ptr.readPointer());
    } catch (e) {
        console.log("Memory access error:", e.message);
    }
}
```

## Conclusion

Frida's HIP runtime injection mechanism provides a powerful and flexible way to trace API calls without modifying the target application. The combination of process control, memory management, and JavaScript execution creates a robust instrumentation platform that can adapt to different runtime environments and function signatures.

The key advantages of this approach are:

1. **Non-invasive**: No source code modification required
2. **Dynamic**: Hooks can be installed and removed at runtime
3. **Flexible**: JavaScript allows for complex tracing logic
4. **Safe**: Memory protection and isolation prevent crashes
5. **Performant**: Optimized for minimal overhead

This mechanism enables comprehensive HIP API tracing while maintaining the stability and performance of the target application.
