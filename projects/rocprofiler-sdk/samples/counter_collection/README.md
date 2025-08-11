# Counter collection

## Services

- Dispatch counting
- Device Counting async
- Device Counting sync
  
## Properties

- Initializes tool and setup for counting service.
- Create a collection profile for the counters.
- Outputs counters mentioned during profiler creation.
- Usage of enum ROCPROFILER_BUFFER_CATEGORY_COUNTERS.
- Buffered_callback
  - This sample shows the usage of buffered approach when collecting counters. buffered callback is called when the buffer is full (or when the buffer is flushed). The callback is responsible for processing the records in the buffer.

- Dispatch callback
  - This sample creates a profile to collect the counter SQ_WAVES for all kernel dispatch packets.
  
- Prints all functional counters.
