// Enhanced Frida script to trace HIP API functions with readable arguments
// Usage: frida -l frida-hip-tracer.js -f /path/to/your/app -- arg1 arg2

console.log("[*] Starting enhanced HIP API tracer with readable arguments");

// Configuration
const OUTPUT_FILE = "/home/aelwazir/work/frida/hip-frida-trace.log";
const CSV_FILE = "/home/aelwazir/work/frida/hip-frida-trace.csv";
const HIP_LIBRARY = "libamdhip64.so.7.1.70100";

// HIP API functions to trace with their argument types
const HIP_FUNCTIONS = {
    "hipMalloc": {
        args: ["void**", "size_t"],
        description: "Allocate memory on device"
    },
    "hipFree": {
        args: ["void*"],
        description: "Free device memory"
    },
    "hipMemcpy": {
        args: ["void*", "const void*", "size_t", "hipMemcpyKind"],
        description: "Copy memory between host and device"
    },
    "hipMemcpyAsync": {
        args: ["void*", "const void*", "size_t", "hipMemcpyKind", "hipStream_t"],
        description: "Asynchronous memory copy"
    },
    "hipLaunchKernel": {
        args: ["const void*", "dim3", "dim3", "void**", "size_t", "hipStream_t"],
        description: "Launch kernel on device"
    },
    "hipLaunchKernelGGL": {
        args: ["const void*", "dim3", "dim3", "void**", "size_t", "hipStream_t"],
        description: "Launch kernel with GGL syntax"
    },
    "hipStreamCreate": {
        args: ["hipStream_t*"],
        description: "Create a stream"
    },
    "hipStreamDestroy": {
        args: ["hipStream_t"],
        description: "Destroy a stream"
    },
    "hipStreamSynchronize": {
        args: ["hipStream_t"],
        description: "Synchronize stream"
    },
    "hipDeviceSynchronize": {
        args: [],
        description: "Synchronize device"
    },
    "hipGetDevice": {
        args: ["int*"],
        description: "Get current device"
    },
    "hipSetDevice": {
        args: ["int"],
        description: "Set current device"
    },
    "hipGetDeviceCount": {
        args: ["int*"],
        description: "Get device count"
    },
    "hipGetDeviceProperties": {
        args: ["hipDeviceProp_t*", "int"],
        description: "Get device properties"
    },
    "hipModuleLoad": {
        args: ["hipModule_t*", "const char*"],
        description: "Load module"
    },
    "hipModuleUnload": {
        args: ["hipModule_t"],
        description: "Unload module"
    },
    "hipModuleGetFunction": {
        args: ["hipFunction_t*", "hipModule_t", "const char*"],
        description: "Get function from module"
    },
    "hipEventCreate": {
        args: ["hipEvent_t*"],
        description: "Create event"
    },
    "hipEventDestroy": {
        args: ["hipEvent_t"],
        description: "Destroy event"
    },
    "hipEventRecord": {
        args: ["hipEvent_t", "hipStream_t"],
        description: "Record event"
    },
    "hipEventSynchronize": {
        args: ["hipEvent_t"],
        description: "Synchronize event"
    },
    "hipGetLastError": {
        args: [],
        description: "Get last error"
    }
};

// Utility functions
function getTimestamp() {
    const now = new Date();
    return now.toISOString().replace('T', ' ').replace('Z', '');
}

function formatPointer(ptr) {
    if (ptr.isNull()) return "NULL";
    return "0x" + ptr.toString(16);
}

function formatValue(value, type, index) {
    if (value === null || value === undefined) return "null";

    try {
        if (type === "void*" || type === "const void*") {
            return formatPointer(value);
        }

        if (type === "void**") {
            if (value.isNull()) return "NULL";
            return formatPointer(value);
        }

        if (type === "size_t") {
            return value.toString();
        }

        if (type === "int" || type === "int*") {
            if (type === "int*" && !value.isNull()) {
                try {
                    const intValue = value.readInt();
                    return `&${intValue}`;
                } catch (e) {
                    return formatPointer(value);
                }
            }
            return value.toString();
        }

        if (type === "hipMemcpyKind") {
            const kinds = {
                0: "hipMemcpyHostToHost",
                1: "hipMemcpyHostToDevice",
                2: "hipMemcpyDeviceToHost",
                3: "hipMemcpyDeviceToDevice",
                4: "hipMemcpyDefault"
            };
            return kinds[value.toInt32()] || `hipMemcpyKind(${value.toInt32()})`;
        }

        if (type === "hipStream_t" || type === "hipStream_t*") {
            if (value.isNull()) return "NULL";
            return formatPointer(value);
        }

        if (type === "hipModule_t" || type === "hipModule_t*") {
            if (value.isNull()) return "NULL";
            return formatPointer(value);
        }

        if (type === "hipFunction_t*") {
            if (value.isNull()) return "NULL";
            return formatPointer(value);
        }

        if (type === "hipEvent_t" || type === "hipEvent_t*") {
            if (value.isNull()) return "NULL";
            return formatPointer(value);
        }

        if (type === "const char*") {
            if (value.isNull()) return "NULL";
            try {
                const str = value.readCString();
                return `"${str}"`;
            } catch (e) {
                return formatPointer(value);
            }
        }

        if (type === "hipDeviceProp_t*") {
            if (value.isNull()) return "NULL";
            return formatPointer(value);
        }

        if (type === "dim3") {
            return `dim3(${value})`;
        }

        // Default: try to read as integer
        if (typeof value.toInt32 === 'function') {
            return value.toInt32().toString();
        }

        return formatPointer(value);

    } catch (e) {
        return formatPointer(value);
    }
}

function logToFile(message) {
    try {
        const file = new File(OUTPUT_FILE, "a");
        file.write(message + '\n');
        file.close();
    } catch (e) {
        console.log("[!] Failed to write to file:", e.message);
    }
}

function logToCSV(timestamp, threadId, functionName, event, args, retval, duration) {
    try {
        const file = new File(CSV_FILE, "a");
        const argsStr = args ? args.join('; ') : '';
        const retStr = retval ? formatPointer(retval) : '';
        const csvLine = `${timestamp},${threadId},${functionName},${event},${argsStr},${retStr},${duration}\n`;
        file.write(csvLine);
        file.close();
    } catch (e) {
        console.log("[!] Failed to write to CSV:", e.message);
    }
}

function createTracer(functionName, funcInfo) {
    try {
        const module = Process.findModuleByName(HIP_LIBRARY);
        if (!module) {
            console.log(`[!] Module ${HIP_LIBRARY} not found`);
            return;
        }

        const exports = module.enumerateExports();
        const export_func = exports.find(function(exp) {
            return exp.name === functionName;
        });

        if (!export_func) {
            console.log(`[!] Function ${functionName} not found`);
            return;
        }

        console.log(`[+] Hooking ${functionName} at ${formatPointer(export_func.address)}`);

        Interceptor.attach(export_func.address, {
            onEnter: function(args) {
                const timestamp = getTimestamp();
                const threadId = Process.getCurrentThreadId();

                let logMessage = `[${timestamp}] [TID:${threadId}] ENTER ${functionName}()`;

                // Log arguments with types
                const argList = [];
                const csvArgs = [];
                try {
                    for (let i = 0; i < funcInfo.args.length && i < 4; i++) {
                        if (args[i] !== undefined && args[i] !== null) {
                            const formattedValue = formatValue(args[i], funcInfo.args[i], i);
                            argList.push(`${funcInfo.args[i]}=${formattedValue}`);
                            csvArgs.push(`${funcInfo.args[i]}=${formattedValue}`);
                        }
                    }
                } catch (e) {
                    // Skip argument logging if there's an error
                }

                if (argList.length > 0) {
                    logMessage += ` [${argList.join(', ')}]`;
                }

                logMessage += ` // ${funcInfo.description}`;

                logToFile(logMessage);
                logToCSV(timestamp, threadId, functionName, "ENTER", csvArgs, null, 0);
                console.log(logMessage);
                this.startTime = Date.now();
                this.csvArgs = csvArgs;
            },

            onLeave: function(retval) {
                const timestamp = getTimestamp();
                const threadId = Process.getCurrentThreadId();
                const duration = this.startTime ? Date.now() - this.startTime : 0;

                let logMessage = `[${timestamp}] [TID:${threadId}] LEAVE ${functionName}() [ret=${formatPointer(retval)}] [duration=${duration}ms]`;

                logToFile(logMessage);
                logToCSV(timestamp, threadId, functionName, "LEAVE", this.csvArgs || [], retval, duration);
                console.log(logMessage);
            }
        });

    } catch (e) {
        console.log(`[!] Failed to hook ${functionName}: ${e.message}`);
    }
}

// Main execution
console.log("[*] Initializing enhanced HIP API tracer...");
console.log(`[*] Output file: ${OUTPUT_FILE}`);
console.log(`[*] CSV file: ${CSV_FILE}`);
console.log(`[*] HIP Library: ${HIP_LIBRARY}`);
console.log(`[*] Tracing ${Object.keys(HIP_FUNCTIONS).length} HIP functions`);

// Initialize output files
try {
    const file = new File(OUTPUT_FILE, "w");
    file.write(`# Enhanced HIP API Trace - Started at ${getTimestamp()}\n`);
    file.write(`# Application: ${Process.getCurrentProcess().name}\n`);
    file.write(`# PID: ${Process.getCurrentProcess().id}\n`);
    file.write(`# Arguments: ${Process.argv.join(' ')}\n\n`);
    file.close();
    console.log("[+] Output file initialized successfully");
} catch (e) {
    console.log("[!] Failed to initialize output file:", e.message);
}

try {
    const csvFile = new File(CSV_FILE, "w");
    csvFile.write("timestamp,thread_id,function_name,event,arguments,return_value,duration_ms\n");
    csvFile.close();
    console.log("[+] CSV file initialized successfully");
} catch (e) {
    console.log("[!] Failed to initialize CSV file:", e.message);
}

// Hook all HIP functions
for (const [funcName, funcInfo] of Object.entries(HIP_FUNCTIONS)) {
    createTracer(funcName, funcInfo);
}

console.log("[*] Enhanced HIP API tracer initialized successfully");
console.log("[*] All HIP function calls will be logged with readable arguments");