## Context
### Current collection architecture
- Collection mode: rocpd

```mermaid
sequenceDiagram
    participant computeLauncher as [rocprof-compute]<br>[Profile phase]<br>utils_profile.py
    participant computeMerger as [rocprof-compute]<br>[Profile phase]<br>rocpd_data.py
    participant process as Target Process
    participant sdkTool as rocprofiler-sdk-tool.so
    participant computeTool as rocprofiler-compute-tool.so
    participant countersWriter as [Interface]<br> Counters Writer
    participant countersWriterCsv as [Impl]<br>Csv Counters Writer
    participant sdkData as [Storage: SQL]<br>[per-process & per-pass]<br>Kernels Data
    participant countersData as [Storage: CSV]<br>[per-process & per-pass]<br>Counters Data
    participant resultDb as [Storage:SQL]<br>[per-pass]<br>Result
    participant resultCsv as [Storage:CSV]<br>[per-pass]<br>Result
    participant computeAnalyze as [rocprof-compute]<br>[Analyze phase]<br>analysis_base.py

    loop each collection pass
        computeLauncher->>process: Launches with LD_PRELOAD of tracing libs
        process->>sdkTool: Loads via LD_PRELOAD
        process->>computeTool: Loads via LD_PRELOAD

        rect rgb(230, 230, 230)
        note right of sdkTool: In-process data collection
            sdkTool->>sdkData: Write
            computeTool->>countersWriter: Pass data
            countersWriter->>countersWriterCsv: Delegate
            countersWriterCsv->>countersData: Write
        end

        computeLauncher->>computeMerger: Requests Merge
        rect rgb(230, 230, 230)
        note right of computeMerger: Data merge
            loop each per-process counters CSV
                computeMerger->>countersData: Read counters
                countersData-->>computeMerger: Counters Data
                computeMerger->>sdkData: Inject counters data to existing per-process SDK databases
            end
            loop each per-process SDK database
                computeMerger->>sdkData: Read counters collection view from database
                computeMerger->>resultDb: Merge data
            end
        end

        computeLauncher->>computeMerger: Request CSV conversion
        rect rgb(230, 230, 230)
        note right of computeMerger: Conversion of data to CSV
            computeMerger->>resultDb: Read data
            resultDb-->>computeMerger: Data
            computeMerger->>resultCsv: Write data to CSV
        end

        rect rgb(230, 230, 230)
        note right of computeLauncher: Removal of intermediate<br>data
            computeLauncher->>sdkData: Deletes per-process rocpds
            computeLauncher->>countersData: Deletes per-process CSVs
            computeLauncher->>resultDb: Deletes result database
        end
    end

    rect rgb(230, 230, 230)
    note left of computeAnalyze: Data read in analysis phase
        loop each per-pass result
            computeAnalyze->>resultCsv: Read
            resultCsv-->>computeAnalyze: Data
            computeAnalyze->>computeAnalyze: merge data to<br>pandas dataframe
        end
    end
```

## Design
### Addition of compression at CSV read/write
```mermaid
sequenceDiagram
    participant computeLauncher as [rocprof-compute]<br>[Profile phase]<br>utils_profile.py
    participant computeMerger as [rocprof-compute]<br>[Profile phase]<br>rocpd_data.py
    participant process as Target Process
    participant sdkTool as rocprofiler-sdk-tool.so
    participant computeTool as rocprofiler-compute-tool.so
    participant countersWriter as [Interface]<br> Counters Writer
    participant countersWriterCsv as [Impl]<br>Csv Counters Writer
    participant sdkData as [Storage: SQL]<br>[per-process & per-pass]<br>Kernels Data
    participant countersData as [Storage: CSV]<br>[per-process & per-pass]<br>Counters Data
    participant resultDb as [Storage:SQL]<br>[per-pass]<br>Result
    participant gzipProfile as Compression lib: gzip (on profile)
    participant resultCsv as [Storage:Compressed CSV]<br>[per-pass]<br>Result
    participant gzipAnalyze as Compression lib: gzip (on analyze)
    participant computeAnalyze as [rocprof-compute]<br>[Analyze phase]<br>analysis_base.py

    loop each collection pass
        computeLauncher->>process: Launches with LD_PRELOAD of tracing libs
        process->>sdkTool: Loads via LD_PRELOAD
        process->>computeTool: Loads via LD_PRELOAD

        rect rgb(230, 230, 230)
        note right of sdkTool: In-process data collection
            sdkTool->>sdkData: Write
            computeTool->>countersWriter: Pass data
            countersWriter->>countersWriterCsv: Delegate
            countersWriterCsv->>countersData: Write
        end

        computeLauncher->>computeMerger: Requests Merge
        rect rgb(230, 230, 230)
        note right of computeMerger: Data merge
            loop each per-process SDK database
                computeMerger->>sdkData: Read database
                sdkData-->>computeMerger: Kernels Data
                computeMerger->>resultDb: Add kernels data to result database
            end
            loop each per-process counters CSV
                computeMerger->>countersData: Read counters
                countersData-->>computeMerger: Counters Data
                computeMerger->>resultDb: Add counters data to result database
            end
        end

        computeLauncher->>computeMerger: Request CSV conversion
        rect rgb(138, 185, 142)
        note right of computeMerger: Conversion of data to CSV<br>with compression
            computeMerger->>resultDb: Read data
            resultDb-->>computeMerger: Data
            computeMerger->>gzipProfile: Write data
            gzipProfile->>resultCsv: Write data to CSV in compressed form
        end

        rect rgb(230, 230, 230)
        note right of computeLauncher: Removal of intermediate<br>data
            computeLauncher->>sdkData: Deletes per-process rocpds
            computeLauncher->>countersData: Deletes per-process CSVs
            computeLauncher->>resultDb: Deletes result database
        end
    end
    rect rgb(138, 185, 142)
    note left of computeAnalyze: Data read in analysis phase
        loop each per-pass result
            computeAnalyze->>gzipAnalyze: Read data
            gzipAnalyze->>resultCsv: Read and uncompress
            resultCsv-->>gzipAnalyze: Data
            gzipAnalyze-->>computeAnalyze: Data
            computeAnalyze->>computeAnalyze: merge data to<br>pandas dataframe
        end
    end

```

### Removal of CSV conversion at the end
- Results from SDK and native collector are processed independently on all phases: profiling and analysis. This is so that we can implement different architectural decisions for them, ex. Profiler Hub integration.
- Do we actually need to have CSV data merge?
    - It adds certain complexity and unknown performance hit.
    - Profiling scripts have to know how to read back the data in order to merge it.

```mermaid
sequenceDiagram
    participant computeLauncher as [rocprof-compute]<br>[Profile phase]<br>utils_profile.py
    participant computeMerger as [rocprof-compute]<br>[Profile phase]<br>rocpd_data.py
    participant process as Target Process
    participant sdkTool as rocprofiler-sdk-tool.so
    participant computeTool as rocprofiler-compute-tool.so
    participant countersWriter as [Interface]<br> Counters Writer
    participant countersWriterCsv as [Impl]<br>Csv Counters Writer
    participant gzipProfile as Compression lib: gzip (on profile)
    participant sdkData as [Storage: SQL]<br>[per-process & per-pass]<br>Kernels Data
    participant countersData as [Storage: Compressed CSV]<br>[per-process & per-pass]<br>Counters Data
    participant resultDb as [Storage:SQL]<br>[per-pass]<br>Result
    participant resultCsv as [Storage:Compressed CSV]<br>[per-pass]<br>Result
    participant gzipAnalyze as Compression lib: gzip (on analyze)
    participant computeAnalyze as [rocprof-compute]<br>[Analyze phase]<br>analysis_base.py

    loop each collection pass
        computeLauncher->>process: Launches with LD_PRELOAD of tracing libs
        process->>sdkTool: Loads via LD_PRELOAD
        process->>computeTool: Loads via LD_PRELOAD

        rect rgb(138, 185, 142)
        note right of sdkTool: In-process data collection
            sdkTool->>sdkData: Write
            computeTool->>countersWriter: Pass data
            countersWriter->>countersWriterCsv: Delegate
            countersWriterCsv->>gzipProfile: Write
            gzipProfile->>countersData: Write compressed data
        end

        computeLauncher->>computeMerger: Requests Merge
        rect rgb(230, 230, 230)
        note right of computeMerger: SDK data merge
            loop each per-process SDK database
                computeMerger->>sdkData: Read database
                sdkData-->>computeMerger: Kernels Data
                computeMerger->>resultDb: Add kernels data to result database
            end
        end
        rect rgb(138, 185, 142)
        note right of computeMerger: CSV data merge
            loop each per-process counters CSV
                computeMerger->>gzipProfile: Read counters
                gzipProfile->>countersData: Read counters
                countersData-->>gzipProfile: Counters Data
                gzipProfile-->>computeMerger: Counters Data
                computeMerger->>gzipProfile: Write counters data
                gzipProfile->>resultCsv: Write counters data
            end
        end

        rect rgb(230, 230, 230)
        note right of computeLauncher: Removal of intermediate<br>data
            computeLauncher->>sdkData: Deletes per-process rocpds
            computeLauncher->>countersData: Deletes per-process CSVs
        end
    end
    rect rgb(138, 185, 142)
    note left of computeAnalyze: Data read in analysis phase
        loop each per-pass result
            computeAnalyze->>resultDb: Read kernel data
            resultDb->>computeAnalyze: Kernel data
            computeAnalyze->>gzipAnalyze: Read counters data
            gzipAnalyze->>resultCsv: Uncompress
            resultCsv-->>gzipAnalyze: Data
            gzipAnalyze-->>computeAnalyze: Data
            computeAnalyze->>computeAnalyze: merge data to<br>pandas dataframe
        end
    end
```

