# hosttrace: application tracing with static/dynamic binary instrumentation and perfetto


## 1. load necessary modules
```
module load julia
module load dyninst
module load rocm/VERSION
```

## 2. to install julia packages, only need it the first time.
```
$ julia
type ']' to get into pkg mode, then 
(@v1.6) pkg> add JSON, DataFrames, Dates, CSV, Chain, PrettyTables
```

## 3. hosttrace usage
```
export PATH=$PATH:$HTRACE_PATH/bin
hosttrace --help
```

## 4. instrument the binaries
```
hosttrace -L $HTRACE_PATH/bin/libhosttrace.so -o app.inst -- path_to_your_app

hosttrace -L $HTRACE_PATH/bin/libhosttrace.so -E 'hipApiName|hipGetCmdName' -o libamdhip64.so.4 --  /opt/rocm-VERSION/lib/libamdhip64.so.4

hosttrace -L $HTRACE_PATH/bin/libhosttrace.so -E 'rocr::atomic|rocr::core|rocr::HSA' -o libhsa-runtime64.so.1 --  /opt/rocm-VERSION/lib/libhsa-runtime64.so.1
```
## 5. run the app
```
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
rocprof --hip-trace --roctx-trace --stats ./app.inst
```

## 6. you may need to increase the buffer size if the hosttrace.perfetto-trace file is close to 1GB. for example to set it to 10GB. 
```
export HOSTTRACE_BUFFER_SIZE_KB=10240000
```

## 7. merge the traces from rocprof and hosttrace
```
julia $HTRACE_PATH/bin/merge-trace.jl results.json hosttrace.perfetto-trace*
```

## 8. another mode of Perfetto tracing is to use system backend. To do it:

### in a separate window
```
    pkill traced; traced --background; perfetto --out ./htrace.out --txt -c $HTRACE/roctrace.cfg
```

### then in the app running window do this before running rocprof or other app
```
    export HOSTTRACE_BACKEND_SYSTEM=1
```

### for the merge use the htrace.out
```
julia $HTRACE_PATH/bin/merge-trace.jl results.json htrace.out
```