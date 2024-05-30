# PC Sampling Method

PC Sampling is a profiling method that uses statistical approximation of the kernel execution by sampling GPU program counters. Furthermore, the method periodically chooses an active wave (in a round robin manner) and snapshot it's program counter (PC). The process takes place on every compute unit simultaneously which makes it device-wide PC sampling. The outcome is the histogram of samples that says how many times each kernel instruction was sampled.

**Note**: The PC sampling feature is still under development and may not be completely stable.

 **Risk Acknowledgment**:
 
  - By activating this feature through `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` environment variable, you acknowledge and accept the following potential risks:
     
     - **Hardware Freeze**: This beta feature could cause your hardware to freeze unexpectedly.
     - **Need for Cold Restart**: In the event of a hardware freeze, you may need to perform a cold restart (turning the hardware off and on) to restore normal operations.
        
 Please use this beta feature cautiously. It may affect your system's stability and performance. Proceed at your own risk.
