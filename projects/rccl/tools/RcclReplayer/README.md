# Structured Logging
As part of the efforts to enhance RCCL Replayer functionality, RCCL now provides detailed logging of API calls.

## Usage
* Structured logging is a built-in module of RCCL source. For RCCL library in ROCm release, it is present starting from ROCm 7.0. To enable structured logging, point LD_LIBRARY_PATH to supporting RCCL library, then run with environment variable `RCCL_REPLAY_FILE="${filename}"`.
* If the value of `RCCL_REPLAY_FILE` contains “.json” extension, the log will be exported as text in **JSON** format. Otherwise, the log will be a **binary** file by default, with ".bin" extension.
* Each process will export its own log local to the directory and node of the executable. Names of each process' output will be in the format of `filename.PID.hostname.extension`. For example, run workload with `RCCL_REPLAY_FILE="replayer_log"` will produce logs named such as `replayer_log.1275.quanta-cyxtera-cx77-11.bin`.
* Log level is controlled by `RCCL_LOG_LEVEL` environment. Currently, by default, log level is 1 and will record most essential RCCL APIs which perform actual operations or incur changes to the communicator. Otherwise, other informational RCCL calls such as `ncclGetAsyncError` or third party function calls like `mscclRunAlgo` will be recorded, too.

## What it does
-   Every public RCCL API call is tracked and logged in order of execution
-   Whenever possible, a log entry is flushed to file immediately after a call is made. Exceptions are when a function returns a handle, in which case the RCCL API may not be registered in case of deadlock or error.
- HIP Graph compatible - will report if a call is made in graph capture mode. If in capture mode, RCCL will append a CPU callback node to the graph which logs the call when the graph runs.

## Output format
Each line in the log is an entry of RCCL API call. `ncclGroupStart()` and 	`ncclGroupEnd()` correspond to opening and closing brackets, each on a separate line, forming a scope. Each line has 2 space of indentation per level of group depth. 

<!---(As a result there may be empty scopes following certain RCCL calls such as Send, Recv, AllToAll, etc., indicating group start/end in its implementation)--->

Each entry will take the format of
`Name : [Parameters : Value, ..., context : [...]](, Trailing Data)`
### Parameters
will contain all the parameters and their values of the function call as defined in nccl.h header. Collectives will contain additional data about the communicator size, number of tasks, its rank, and its opCount in communicator. 

Wherever applicable, the structured logging preserves underlying RCCL data constructs and how they are filled.

<!---We try to register and flush logging information at the beginning of a function, in case it never completes before termination/hanging of the program. **However**, many RCCL routines, such as communicator creation, user buffer registration, etc. will have pointers for returned handles. We record those value as well, but at the end of the routine, therefore these calls may not be logged in face of deadlock or error.---> 

<!---Please interpret the parameters with a grain of salt. They are logged exactly as they are used, by user or by NCCL internal implementations. For instance, `ncclSend` entries will always have a null sendbuff but a valid "recvbuff" in the log, as `ncclSend` under the hood always fills the send buffer into the recv buffer field of `ncclInfo` that is enqueued.--->
 

### Device context
contains the following fields: `timestamp`, `thread` (caller thread ID), `device` ( GPU ID which the caller was bound to), `captured` (graph capture mode or not), and the `graphID`. If a call was made in graph capture mode and not actually running on GPU device, `captured` will be 1. All entries would have a `graphID`, but their validity depend on whether there were previous captures.
### Trailing Info
Certain calls will have additional data following the parameters and contexts:
* `ncclCommInitAll` will print of list of devices
* `ncclCommInitRankConfig` and `ncclCommSplit` will have fields of `ncclConfig_t`
* `ncclGroupSimulateEnd` will print all fields of `ncclSimInfo_t`
* `ncclAllToAllv` will be followed by four comma separated lists of send/recv counts and displacements


## Example
Here is an example log of the `CommSplit_Reduces` Unit Test from RCCL, run with `RCCL_REPLAY_FILE="log.json" UT_DATATYPES="ncclInt32" UT_PROCESS_MASK=2 UT_MIN_GPUS=4 ./rccl-UnitTests --gtest_filter="Standalone.SplitComms_Reduce" UT_DATATYPES="ncclInt32"`
```
{
  hostname : banff-pla-r27-29, version : 0,
  CommInitAll : [# of device : 4, context : [time : 1745623615674.781006, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  GetUniqueId : [uniqueID : 17168916771200794912, context : [time : 1745623615676.041748, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  {
    CommInitDev : [comm : 0x7fb680328010, size : 4, uniqueID : 17168916771200794912, rank : 0, dev : 0, context : [time : 1745623615676.157959, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
    CommInitDev : [comm : 0x7fa67bb28010, size : 4, uniqueID : 17168916771200794912, rank : 1, dev : 1, context : [time : 1745623615676.207520, thread : 1085917, device : 1, captured : -1, graphID : 0 ]],
    CommInitDev : [comm : 0x7f967b650010, size : 4, uniqueID : 17168916771200794912, rank : 6, dev : 6, context : [time : 1745623615676.407227, thread : 1085917, device : 6, captured : -1, graphID : 0 ]],
    CommInitDev : [comm : 0x7f8e7ad27010, size : 4, uniqueID : 17168916771200794912, rank : 7, dev : 7, context : [time : 1745623615676.468750, thread : 1085917, device : 7, captured : -1, graphID : 0 ]]
  },
  {
    CommSplit : [comm : 0x7fb680328010, color : 0, key : 0, newcomm : (nil), context : [time : 1745623619986.908691, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
    CommSplit : [comm : 0x7fa67bb28010, color : 0, key : 0, newcomm : 0x1, context : [time : 1745623619987.017090, thread : 1085917, device : 1, captured : -1, graphID : 0 ]],
    CommSplit : [comm : 0x7f967b650010, color : 0, key : -1, newcomm : 0x6, context : [time : 1745623619987.202393, thread : 1085917, device : 6, captured : -1, graphID : 0 ]],
    CommSplit : [comm : 0x7f8e7ad27010, color : 0, key : -1, newcomm : 0x7, context : [time : 1745623619987.205322, thread : 1085917, device : 7, captured : -1, graphID : 0 ]]
  },
  CommDestroy : [comm : 0x126d9a0, context : [time : 1745623620419.313721, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : 0x1345370, context : [time : 1745623620419.422363, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : (nil), context : [time : 1745623620888.018555, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : (nil), context : [time : 1745623620888.020996, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : 0x7fb680328010, context : [time : 1745623620888.023193, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : 0x7fa67bb28010, context : [time : 1745623620888.228027, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : 0x7f967b650010, context : [time : 1745623620889.099121, thread : 1085917, device : 0, captured : -1, graphID : 0 ]],
  CommDestroy : [comm : 0x7f8e7ad27010, context : [time : 1745623620889.293213, thread : 1085917, device : 0, captured : -1, graphID : 0 ]]
}
```
# RCCL Replayer
Replayer is a separate tool which aims to re-run the same set of RCCL calls as recorded and report the cumulative time taken by these calls, provided with the **binary** output of structured logging.
## Installation
* Replayer relies on MPI for out of band communication.
* Under `rccl/tools/RcclReplayer`, run `MPI_DIR=${MPI_PATH} make`
*  Replayer can be built from RCCL source using internal headers directly. It links against a RCCL library specified by `RCCL_DIR` (defaults to `../../build/release`). For compatibility, it is recommended that logs are collected using the same RCCL library version.
## Running
* Replayer requires the exact same number of processes and processes per node as the recorded job. And all log files must be accessible by all processes in Replayer, either through shared filesystem or copies.
* To run Replayer, simply call `mpirun -np ${np} ./rcclReplayer ${filename}.${extension}` 
* For example, we are on node quanta-cyxtera-cx77-11, with 8 logs: `replayer_log.{1270-1278}.quanta-cyxtera-cx77-11.bin`. Run `mpirun -np 8 ./rcclReplayer replayer_log.bin`
* Replayer will have a parse and replay phase. During parsing it will create communicators as original RCCL job, assign log files to individual processes, and allocate resources. Then actual replay happens, re-running all the RCCL APIs with same parameters and device assignment. It is also able to capture and launch graphs involving RCCL calls, as recorded by structured logging. Actual data in original job such as message payload or vector values are not recorded therefore not replicated.


## Output
Each rank will print out its progress as it goes through every line of calls, including its rank, line number, RCCL API name, status (INFO/WARNING/ERROR). 
It will also report time and bandwidth (if the line is a communication call) for that call. In the end, it will report the total time taken by all communication calls.
Replayer is still under development and experimentations, so the formats of logging or contents of replayer output will be subject to changes.
