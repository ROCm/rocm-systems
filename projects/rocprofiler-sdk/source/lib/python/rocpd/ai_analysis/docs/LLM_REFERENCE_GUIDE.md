# LLM Reference Guide Documentation

## How to Customize LLM Behavior Without Changing Code

**Version:** 1.0.0

---

## Overview

The **LLM Reference Guide** (also called the "fence") is a user-modifiable markdown file that controls how the LLM analyzes GPU profiling data. It acts as a constraint system that ensures the LLM provides high-quality, actionable insights following AMD GPU best practices.

**Key Concept:** The reference guide is loaded at runtime and included in every LLM request as part of the system prompt. This means you can modify LLM behavior by simply editing a text file - **no code changes required**.

---

## Table of Contents

1. [What is the Reference Guide?](#what-is-the-reference-guide)
2. [Where is it Located?](#where-is-it-located)
3. [How It Works](#how-it-works)
4. [Common Modifications](#common-modifications)
5. [Best Practices](#best-practices)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## What is the Reference Guide?

The reference guide (`llm-reference-guide.md`) contains:

1. **GPU Hardware Specifications**
   - AMD GPU architectures (MI100, MI250X, MI300X)
   - Peak FLOPS, memory bandwidth, compute units
   - Cache sizes, wave characteristics

2. **Performance Analysis Models**
   - Roofline model formulas
   - Speed-of-Light (SOL) metrics
   - Top-Down analysis methodology

3. **Bottleneck Classification Rules**
   - Compute-bound indicators
   - Memory-bound indicators
   - Latency-bound indicators

4. **AMD-Specific Optimization Techniques**
   - Wave occupancy optimization
   - LDS (Local Data Share) usage
   - Memory coalescing patterns
   - MFMA instructions for matrix ops

5. **Recommendation Quality Standards**
   - Structure requirements (title, priority, description, actionable steps)
   - Prioritization criteria (high/medium/low)
   - Example good vs bad recommendations

6. **Output Format Requirements**
   - Report structure (summary, breakdown, recommendations)
   - Tone and style guidelines

7. **Prohibited Actions**
   - What the LLM should NOT do
   - Privacy and security boundaries

---

## Where is it Located?

### Default Locations (in priority order):

1. **Environment Variable** (highest priority):
   ```bash
   export ROCPD_LLM_REFERENCE_GUIDE=/path/to/custom/guide.md
   ```

2. **ROCm Installation Directory**:
   ```
   /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md
   ```

3. **Development Source Directory**:
   ```
   rocm-systems-dev/projects/rocprofiler-sdk/share/llm-reference-guide.md
   ```

### How to Find Your Reference Guide:

```bash
# Check which location is being used
python3 -c "from rocpd.ai_analysis.llm_analyzer import get_reference_guide_path; print(get_reference_guide_path())"
```

---

## How It Works

### The Flow:

```
User runs: rocpd analyze --ai output.db --llm anthropic

↓

1. LLMAnalyzer.__init__() is called
   - Loads llm-reference-guide.md from default location
   - Stores content in self.reference_guide

↓

2. analyze_with_llm() is called
   - Sanitizes profiling data
   - Builds system prompt: "You are an expert. [REFERENCE GUIDE CONTENT]"
   - Sends to Anthropic Claude API with sanitized data

↓

3. LLM receives:
   System Prompt: [Reference guide with all GPU specs, models, guidelines]
   User Prompt: [Sanitized profiling data]

↓

4. LLM generates analysis following the reference guide

↓

5. Natural language explanation returned to user
```

### Why This Design?

- ✅ **Easy to modify** - Edit text file, no code changes
- ✅ **Version control** - Reference guide can be tracked in git
- ✅ **Team customization** - Different teams can use different guides
- ✅ **No recompilation** - Changes take effect immediately
- ✅ **Transparent** - Clear what the LLM knows and doesn't know

---

## Common Modifications

### 1. Add New GPU Architecture

**Scenario:** AMD releases MI400 (gfx950) and you want the LLM to analyze MI400 traces.

**Steps:**

1. Open the reference guide:
   ```bash
   sudo nano /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md
   ```

2. Add new GPU section:
   ```markdown
   ### MI400 (gfx950)
   - **Architecture**: CDNA 4
   - **Compute Units**: 400
   - **Peak FP64**: 300 TFLOPS
   - **Peak FP32**: 300 TFLOPS
   - **Peak FP16**: 1200 TFLOPS
   - **Memory**: 256 GB HBM3e
   - **Memory Bandwidth**: 8.0 TB/s
   - **L2 Cache**: 512 MB
   - **Wave Size**: 64 threads
   - **Max VGPRs per Wave**: 256
   - **LDS per CU**: 128 KB
   ```

3. Save and exit

4. **Done!** Next analysis will use the updated specs:
   ```bash
   rocpd analyze --ai mi400_trace.db --llm anthropic
   ```

**No code changes, no recompilation required.**

---

### 2. Change Priority Thresholds

**Scenario:** Your team considers anything impacting >5% of runtime as high priority (default is >10%).

**Steps:**

1. Open reference guide

2. Find the "Recommendation Quality Guidelines" section

3. Modify:
   ```markdown
   ### 9.2 Prioritization
   Recommendations must be ranked:
   - **High Priority**: Impacts >5% of total execution time (was >10%)
   - **Medium Priority**: Impacts 2-5% of total execution time (was 3-10%)
   - **Low Priority**: Impacts <2% but still worth addressing
   ```

4. Save

**Result:** LLM will now classify more recommendations as high priority.

---

### 3. Add Company-Specific Guidelines

**Scenario:** Your organization has specific performance requirements or standards.

**Steps:**

1. Open reference guide

2. Add new section before "Summary":
   ```markdown
   ## Company-Specific Requirements

   ### Power Consumption Standards
   - Always mention power consumption if data available
   - Target: <300W per GPU for datacenter deployments
   - Flag kernels with >350W as "High Power"

   ### Multi-GPU Scaling
   - For multi-GPU workloads, always analyze scaling efficiency
   - Target: >90% scaling efficiency for 2-GPU
   - Target: >80% scaling efficiency for 4-GPU

   ### Naming Conventions
   - Use our internal kernel naming conventions:
     - compute_* kernels → Primary computation
     - transfer_* kernels → Data movement
     - sync_* kernels → Synchronization

   ### Approval Requirements
   - All recommendations must reference internal optimization guide:
     "See AMD GPU Optimization Guide (Internal Doc #12345)"
   ```

3. Save

**Result:** LLM will follow your company's standards when generating recommendations.

---

### 4. Update Optimization Techniques

**Scenario:** New AMD optimization technique discovered (e.g., Wave32 mode for latency-sensitive kernels).

**Steps:**

1. Open reference guide

2. Find "AMD-Specific Optimization Techniques" section

3. Add new subsection:
   ```markdown
   ### 5. Wave32 Mode (New in ROCm 6.3)
   **When**: Latency-sensitive kernels with low arithmetic intensity

   **Benefits**:
   - Lower latency (2x faster wave launch)
   - Better for small workloads
   - Improved cache utilization

   **How to Enable**:
   ```c
   __attribute__((amdgpu_waves_per_eu(2,4)))
   __attribute__((amdgpu_flat_work_group_size(32,32)))
   __global__ void my_kernel() { ... }
   ```

   **When NOT to use**:
   - High compute intensity kernels
   - Kernels bottlenecked by memory bandwidth
   ```

4. Save

**Result:** LLM will recommend Wave32 mode when appropriate.

---

### 5. Modify Tone and Style

**Scenario:** You want more technical depth or less jargon.

**Steps:**

1. Open reference guide

2. Find "Output Format Requirements" section

3. Modify tone requirements:
   ```markdown
   ### Tone:
   - Extremely technical - assume expert audience
   - Use precise GPU terminology (no simplifications)
   - Include ISA-level details when relevant
   - Cite specific GCN/CDNA architecture manuals
   - Provide assembly code examples where helpful
   ```

   OR for less technical:
   ```markdown
   ### Tone:
   - Simple and accessible - assume beginner audience
   - Avoid jargon, use analogies
   - Focus on "what" and "why", not "how"
   - Provide step-by-step instructions
   - Use layman's terms (avoid ISA, wavefronts, etc.)
   ```

4. Save

**Result:** LLM adjusts complexity level accordingly.

---

### 6. Add New Performance Model

**Scenario:** Your research team developed a new "Cache Efficiency Model" and you want the LLM to use it.

**Steps:**

1. Open reference guide

2. Add new model section:
   ```markdown
   ## Cache Efficiency Model (Custom)

   ### Purpose
   Predict cache hit rates based on data access patterns

   ### Formula
   ```
   Cache Hit Rate = (L2_HIT / (L2_HIT + L2_MISS)) * 100%
   Effective Bandwidth = HBM_Bandwidth * (1 - CacheHitRate)
   ```

   ### Interpretation
   - **>80% hit rate**: Excellent cache locality
   - **50-80% hit rate**: Good, but room for improvement
   - **<50% hit rate**: Poor locality, consider data layout changes

   ### Recommendations Based on This Model
   - Low hit rate + memory-bound → Improve data layout (SoA vs AoS)
   - Low hit rate + compute-bound → Not a priority
   - High hit rate + memory-bound → Bottleneck is bandwidth, not latency
   ```

3. Save

**Result:** LLM incorporates your custom model in analysis.

---

## Best Practices

### DO ✅

- **Be specific**: Provide concrete formulas, thresholds, and examples
- **Use examples**: Show "good" vs "bad" recommendation examples
- **Version control**: Track changes to the guide in git
- **Test changes**: Run analysis on sample traces after modifying
- **Document changes**: Add comments explaining why you made changes
- **Keep organized**: Maintain clear section structure
- **Use markdown**: Proper formatting helps LLM parse information

### DON'T ❌

- **Don't be vague**: "Optimize memory" → "Reduce VGPR usage below 64 to increase occupancy"
- **Don't contradict**: Make sure all sections are consistent
- **Don't remove critical sections**: GPU specs, performance models are required
- **Don't make it too long**: LLMs have token limits (~200K tokens)
- **Don't include sensitive data**: No internal code, proprietary algorithms
- **Don't break markdown syntax**: LLM uses formatting as structure

---

## Examples

### Example 1: Team-Specific Guide

```markdown
# LLM Reference Guide - HPC Team Configuration

## Team Requirements
- Target hardware: MI300X only
- Focus: Large-scale scientific computing
- Priority: Multi-GPU scaling over single-GPU optimization

## Custom Thresholds
- High priority: >3% of runtime (aggressive optimization)
- Communication overhead: Flag if >15% (was >20%)
- Strong scaling efficiency: Target >85% (was >80%)

... (rest of guide)
```

### Example 2: Machine Learning Focus

```markdown
# LLM Reference Guide - ML/DL Workloads

## Optimization Focus Areas
1. MFMA utilization (matrix operations)
2. Tensor memory layout (NHWC vs NCHW)
3. Mixed precision (FP16/BF16 usage)
4. Batch size optimization

## ML-Specific Bottlenecks
- **Low MFMA utilization** (<60%): Not using matrix instructions
- **High H2D/D2H**: Data pipeline bottleneck
- **Small kernels**: Launch overhead dominates

... (rest of guide)
```

---

## Troubleshooting

### Problem: LLM isn't following my changes

**Possible causes:**

1. **Editing wrong file**
   - Check which file is being loaded:
     ```bash
     python3 -c "from rocpd.ai_analysis.llm_analyzer import get_reference_guide_path; print(get_reference_guide_path())"
     ```

2. **Syntax error in markdown**
   - Validate markdown syntax
   - Check for unclosed code blocks, tables

3. **LLM API issues**
   - Verify API key is valid
   - Check API rate limits

4. **Changes too subtle**
   - LLMs may not always strictly follow guidelines
   - Make changes more explicit

**Solution:**
```bash
# Set explicit path to your guide
export ROCPD_LLM_REFERENCE_GUIDE=/path/to/my/custom-guide.md

# Run analysis with verbose mode
rocpd analyze --ai output.db --llm anthropic --verbose

# Check which guide was loaded
```

---

### Problem: Reference guide not found

**Error:**
```
ReferenceGuideNotFoundError: LLM reference guide not found at: /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md
```

**Solutions:**

1. **Copy from source to installation:**
   ```bash
   sudo cp rocm-systems-dev/projects/rocprofiler-sdk/share/llm-reference-guide.md \
            /opt/rocm/share/rocprofiler-sdk/
   ```

2. **Use environment variable:**
   ```bash
   export ROCPD_LLM_REFERENCE_GUIDE=/path/to/llm-reference-guide.md
   ```

3. **Reinstall rocprofiler-sdk:**
   ```bash
   cd rocm-systems-dev
   cmake --build projects/rocprofiler-sdk/build --target install
   ```

---

### Problem: LLM output quality degraded

**Symptoms:**
- Generic recommendations
- Missing details
- Incorrect GPU specs

**Possible causes:**
1. Accidentally deleted critical sections
2. Markdown formatting broken
3. Contradictory guidelines

**Solution:**
1. Restore original guide from git:
   ```bash
   git checkout share/llm-reference-guide.md
   ```

2. Compare with original:
   ```bash
   diff llm-reference-guide.md /opt/rocm/share/rocprofiler-sdk/llm-reference-guide.md
   ```

3. Revert changes incrementally

---

## Advanced: Custom Reference Guides for Different Use Cases

### Scenario: Multiple teams with different needs

**Setup:**

```bash
# Create team-specific guides
/opt/rocm/share/rocprofiler-sdk/
├── llm-reference-guide.md              # Default
├── llm-reference-guide-hpc.md          # HPC team
├── llm-reference-guide-ml.md           # ML team
├── llm-reference-guide-gaming.md       # Gaming team
```

**Usage:**

```bash
# HPC team
export ROCPD_LLM_REFERENCE_GUIDE=/opt/rocm/share/rocprofiler-sdk/llm-reference-guide-hpc.md
rocpd analyze --ai hpc_trace.db --llm anthropic

# ML team
export ROCPD_LLM_REFERENCE_GUIDE=/opt/rocm/share/rocprofiler-sdk/llm-reference-guide-ml.md
rocpd analyze --ai ml_trace.db --llm anthropic
```

**Benefit:** Each team gets analysis tailored to their domain.

---

## Summary

The LLM Reference Guide is the "fence" that keeps AI analysis high-quality and consistent. By modifying this simple markdown file, you can:

- ✅ Add new GPU architectures
- ✅ Customize priority thresholds
- ✅ Add company-specific guidelines
- ✅ Update optimization techniques
- ✅ Change output style and tone
- ✅ Add custom performance models

**No code changes, no recompilation, just edit the text file.**

---

## See Also

- [AI Analysis API Documentation](AI_ANALYSIS_API.md)
- [rocpd CLI Documentation](../README.md)
- Reference guide source: `share/llm-reference-guide.md`

---

## Support

For issues with the reference guide:
- File an issue on GitHub
- Include your modified guide (if appropriate)
- Specify LLM provider and model used
