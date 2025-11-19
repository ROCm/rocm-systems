/**
 * @file packet_generation.c
 * @brief Implementation of PM4 packet generation functions for counter
 * operations
 */

#include "packet_generation.h"

#ifdef __KERNEL__
#include <linux/errno.h>
#else
#include <errno.h>
#include <stdbool.h>
#endif

#define VALIDATE_BLOCK_ID(arch, counter) \
    (((counter)->block_id >= HW_IP_BLOCK_LAST) || \
     ((counter)->block_id > (arch)->block_map.block_count) || \
     (!(arch)->block_map.blocks[(counter)->block_id]))

/**
 * @brief Generate CS partial flush packet to synchronize compute shader execution
 *
 * Appends an EVENT_WRITE packet that triggers a CS_PARTIAL_FLUSH event, which
 * ensures that all compute shader work preceding this point completes before
 * subsequent commands execute. Used for synchronization in performance counter
 * start/stop/read sequences.
 *
 * This function is analogous to the barrier commands in GpuPmcBuilder::Start() and
 * GpuPmcBuilder::Read() in projects/aqlprofile/src/pm4/pmc_builder.h:240, 507
 *
 * @param buffer PM4 buffer to append packet to
 * @param arch Architecture information containing event configuration
 * @return 0 on success, -EINVAL if buffer or arch is NULL
 *
 * @note Called at critical synchronization points: before starting counters,
 *       before reading counter values, and after stopping counters
 * @see pm4_append_event_write(), GpuPmcBuilder::Start in
 *      projects/aqlprofile/src/pm4/pmc_builder.h:240
 */
int generate_cs_partial_flush(pm4_buffer_t *buffer, const arch_t *arch) {
  if (!buffer || !arch) {
    return -EINVAL;
  }

  return pm4_append_event_write(buffer,
                                arch->control_regs.cs_partial_flush_event,
                                arch->control_regs.event_index_flush);
}

/**
 * @brief Set GRBM to broadcast mode for writing to all GPU instances
 *
 * Configures the GRBM_GFX_INDEX register to broadcast writes to all Shader Engines,
 * Shader Arrays, and WGPs simultaneously. This is essential before configuring
 * performance counters that exist across multiple hardware instances.
 *
 * This function is analogous to GpuPmcBuilder::SetGrbmBroadcast() in
 * projects/aqlprofile/src/pm4/pmc_builder.h:191-193
 *
 * @param buffer PM4 buffer to append packet to
 * @param arch Architecture information with GRBM_GFX_INDEX register offset
 * @return 0 on success, -EINVAL if buffer or arch is NULL
 *
 * @note Must be called before any counter configuration to ensure all instances
 *       receive the same settings
 * @see pm4_grbm_broadcast(), GpuPmcBuilder::SetGrbmBroadcast in
 *      projects/aqlprofile/src/pm4/pmc_builder.h:191
 */
int generate_grbm_broadcast(pm4_buffer_t *buffer, const arch_t *arch) {
  if (!buffer || !arch) {
    return -EINVAL;
  }

  return pm4_grbm_broadcast(buffer, arch->control_regs.grbm_gfx_index);
}

/**
 * @brief Enable or disable performance monitoring with optional sampling
 *
 * Writes to the CP_PERFMON_CNTL register to control the performance monitoring
 * state machine. The perfmon state determines whether counters are running,
 * stopped, or disabled. The sample bit enables reading counter values without
 * stopping them.
 *
 * This function is analogous to GpuPmcBuilder::SetPerfmonCntl() in
 * projects/aqlprofile/src/pm4/pmc_builder.h:195-198
 *
 * @param buffer PM4 buffer to append packet to
 * @param arch Architecture information with CP_PERFMON_CNTL register offset
 * @param enable_state Perfmon state: 0=disable, 1=enable/start, 2=stop
 * @param sample_enable If true, sets the sample bit to enable non-destructive reads
 * @return 0 on success, -EINVAL if buffer or arch is NULL
 *
 * @note The perfmon_state field (bits 0-3) controls the state machine
 * @note The sample bit (typically bit 10) enables reading while counting
 * @see pm4_perfcount_enable(), GpuPmcBuilder::SetPerfmonCntl in
 *      projects/aqlprofile/src/pm4/pmc_builder.h:195
 */
int generate_perfmon_enable(pm4_buffer_t *buffer, const arch_t *arch,
                            uint8_t enable_state, bool sample_enable) {
  if (!buffer || !arch) {
    return -EINVAL;
  }

  /* Build perfmon control value */
  uint32_t perfmon_value = enable_state & 0xF; /* perfmon_state in bits 0-3 */
  if (sample_enable) {
    perfmon_value |=
        (1U << arch->control_regs.perfmon_states.perfmon_sample_bit);
  }

  return pm4_perfcount_enable(buffer, arch->control_regs.cp_perfmon_cntl,
                              perfmon_value);
}

/**
 * @brief Configure counter selection and control registers for a specific counter
 *
 * Writes the event ID to the counter's SELECT register and optionally configures
 * the CONTROL register (for SQ counters, enables all shader stages). This sets up
 * what event the hardware will count.
 *
 * This function is analogous to the counter configuration loop in
 * GpuPmcBuilder::Start() at projects/aqlprofile/src/pm4/pmc_builder.h:284-439
 *
 * @param buffer PM4 buffer to append packets to
 * @param arch Architecture information with block definitions
 * @param counter Counter to configure (block, event ID, counter index)
 * @return 0 on success, -EINVAL for invalid parameters, -ENOENT if block not found
 *
 * @note For SQ (shader) counters, enables all shader stages (PS, GS, HS, CS)
 * @note Validates counter index against block's counter_count
 * @see generate_start_packet(), GpuPmcBuilder::Start in
 *      projects/aqlprofile/src/pm4/pmc_builder.h:284
 */
int generate_counter_config(pm4_buffer_t *buffer, const arch_t *arch,
                            const counter_info_t *counter) {
  int ret;

  if (!buffer || !arch || !counter) {
    return -EINVAL;
  }

  /* Get block info for this counter */
  if (VALIDATE_BLOCK_ID(arch, counter)) {
    return -ENOENT; /* Block not found */
  }

  block_info_t *block = arch->block_map.blocks[counter->block_id];
  /* Validate counter index */
  if (counter->counter_index >= block->counter_count) {
    return -EINVAL; /* Counter index out of range */
  }

  /* Get register info for this counter */
  counter_reg_info_t *reg_info =
      &block->counter_reg_info[counter->counter_index];

  /* Write counter select register */
  uint32_t select_value = counter->event_id & 0x1FF; /* 9-bit event ID */
  ret = pm4_append_set_uconfig_reg(
      buffer, reg_info->select_addr,
      select_value);
  if (ret < 0) {
    return ret;
  }

  /* Write counter control register if present */
  if (reg_info->control_addr != 0) {
    uint32_t control_value = 0;

    /* For SQ counters, enable all shader stages */
    if (counter->block_id == HW_IP_BLOCK_SQ) {
      control_value |=
          (1U << arch->control_regs.counter_control_bits.sq_ps_en_bit); /* PS */
      control_value |=
          (1U << arch->control_regs.counter_control_bits.sq_gs_en_bit); /* GS */
      control_value |=
          (1U << arch->control_regs.counter_control_bits.sq_hs_en_bit); /* HS */
      control_value |=
          (1U << arch->control_regs.counter_control_bits.sq_cs_en_bit); /* CS */
    }

    ret = pm4_append_set_uconfig_reg(
        buffer, reg_info->control_addr,
        control_value);
    if (ret < 0) {
      return ret;
    }
  }

  return 0;
}

/* Generate PM4 packet sequence to start performance counters */
int generate_start_packet(pm4_buffer_t *buffer, const arch_t *arch,
                          const counter_collection_t *collection) {
  int ret;

  if (!buffer || !arch || !collection || !collection->counters) {
    return -EINVAL;
  }

  /* 1. CS partial flush */
  ret = generate_cs_partial_flush(buffer, arch);
  if (ret < 0)
    return ret;

  /* 2. GRBM broadcast mode */
  ret = generate_grbm_broadcast(buffer, arch);
  if (ret < 0)
    return ret;

  /* 3. Disable perfmon initially */
  ret = generate_perfmon_enable(
      buffer, arch, arch->control_regs.perfmon_states.perfmon_state_disable,
      false);
  if (ret < 0)
    return ret;

  /* 4. Enable SQ control for SQ counters (force_en=1, vmid_en=0xFFFF) */
  bool has_sq_counters = false;
  for (size_t i = 0; i < collection->counter_count; i++) {
    if (collection->counters[i].block_id == HW_IP_BLOCK_SQ) {
      has_sq_counters = true;
      break;
    }
  }

  if (has_sq_counters) {
    uint32_t sq_ctrl2_value =
        (1U << 0) | (0xFFFFU << 1); /* force_en | vmid_en */
    ret = pm4_append_set_uconfig_reg(
        buffer,
        arch->control_regs.sq_perfcounter_ctrl2, /* mmSQ_PERFCOUNTER_CTRL2 */
        sq_ctrl2_value);
    if (ret < 0)
      return ret;

    /* Reset GRBM_GFX_INDEX to broadcast before configuring counters */
    ret = generate_grbm_broadcast(buffer, arch);
    if (ret < 0)
      return ret;
  }

  /* 5. Configure each counter in broadcast mode */
  for (size_t i = 0; i < collection->counter_count; i++) {
    counter_info_t *counter = &collection->counters[i];

    /* Configure the counter (broadcast mode already set) */
    ret = generate_counter_config(buffer, arch, counter);
    if (ret < 0)
      return ret;
  }

  /* 6. GRBM broadcast again */
  ret = generate_grbm_broadcast(buffer, arch);
  if (ret < 0)
    return ret;

  /* 7. Enable compute perfcount */
  ret = pm4_append_write_sh_reg(buffer,
                                arch->control_regs.compute_perfcount_enable,
                                0x1, /* enable */
                                0, 0);
  if (ret < 0)
    return ret;

  /* 8. Enable perfmon (disable first, then enable) */
  ret = generate_perfmon_enable(
      buffer, arch, arch->control_regs.perfmon_states.perfmon_state_disable,
      false);
  if (ret < 0)
    return ret;

  ret = generate_perfmon_enable(
      buffer, arch, arch->control_regs.perfmon_states.perfmon_state_enable,
      false);
  if (ret < 0)
    return ret;

  /* 9. Final CS partial flush */
  ret = generate_cs_partial_flush(buffer, arch);
  if (ret < 0)
    return ret;

  return 0;
}

/* Generate PM4 packet sequence to read performance counters */
int generate_read_packet(pm4_buffer_t *buffer, const arch_t *arch,
                         const counter_collection_t *collection) {
  int ret;
  uint64_t current_addr;

  if (!buffer || !arch || !collection || !collection->counters) {
    return -EINVAL;
  }

  /* 1. Enable perfmon with sampling */
  ret = generate_perfmon_enable(
      buffer, arch, arch->control_regs.perfmon_states.perfmon_state_enable,
      true); /* enable sampling */
  if (ret < 0)
    return ret;

  /* 2. GRBM broadcast mode */
  ret = generate_grbm_broadcast(buffer, arch);
  if (ret < 0)
    return ret;

  /* 3. CS partial flush */
  ret = generate_cs_partial_flush(buffer, arch);
  if (ret < 0)
    return ret;

  /* 4. Read counters from all topology locations */
  current_addr = collection->gpu_memory_addr;

  for (size_t counter_idx = 0; counter_idx < collection->counter_count;
       counter_idx++) {
    counter_info_t *counter = &collection->counters[counter_idx];
    block_info_t *block = arch->block_map.blocks[counter->block_id];

    if (!block) {
      return -ENOENT;
    }

    /* Reset to broadcast mode for each counter */
    ret = generate_grbm_broadcast(buffer, arch);
    if (ret < 0)
      return ret;

    /* Get register info */
    counter_reg_info_t *reg_info =
        &block->counter_reg_info[counter->counter_index];

    /*
     * Always iterate through GPU topology based on block dimensions.
     * This reads all instances; software filtering will select specific
     * dimensions if needed.
     */
    bool has_se_dimension = false;
    uint32_t num_se = arch->num_se;
    uint32_t num_sa = arch->num_sa;
    uint32_t num_wgp = arch->num_wgp_per_sa;

    /* Extract dimension sizes from block dimensions */
    for (size_t dim_idx = 0; dim_idx < block->dimension_count; dim_idx++) {
      dimension_t *dim = &block->dimensions[dim_idx];
      if (dim->dim == HARDWARE_DIM_SE) {
        num_se = dim->size;
        has_se_dimension = true;
      } else if (dim->dim == HARDWARE_DIM_SA) {
        num_sa = dim->size;
      } else if (dim->dim == HARDWARE_DIM_WGP) {
        num_wgp = dim->size;
      }
    }

    if (has_se_dimension) {
      /*
       * SE-dependent block - iterate through SE x SA x WGP using block-specific dimensions.
       *
       * Read Strategy Rationale:
       * We always read ALL instances from the GPU (broadcast mode) and perform software
       * filtering to return specific dimensions when requested. This approach is preferred
       * because:
       *
       * 1. Hardware Consistency: GRBM_GFX_INDEX SE-level filtering is well-tested and
       *    reliable across AMD GPU generations. Finer-grained WGP/CU filtering via
       *    instance_index has hardware quirks and varies by architecture.
       *
       * 2. Reference Implementation: This matches the approach used in aqlprofiler
       *    (projects/aqlprofile), which successfully uses SE-level filtering with
       *    software aggregation for production workloads.
       *
       * 3. Simplicity: Using broadcast mode for reads and filtering in software keeps
       *    the PM4 packet generation straightforward and reduces hardware-specific edge
       *    cases.
       *
       * 4. Performance: For typical perf use cases reading a small number of counters,
       *    the overhead of reading all instances is negligible compared to GPU execution
       *    and kernel overhead.
       *
       * Software filtering in aql_perf_measurement_read() handles dimension-specific
       * requests by indexing into the results array using encode_dimension_index().
       */
      for (uint32_t se = 0; se < num_se; se++) {
        for (uint32_t sa = 0; sa < num_sa; sa++) {
          for (uint32_t wgp = 0; wgp < num_wgp; wgp++) {
            /* Set GRBM index for specific location */
            ret =
                pm4_set_grbm_index(buffer, arch->control_regs.grbm_gfx_index,
                                   wgp, /* instance_index - will be shifted in pm4_set_grbm_index */
                                   sa, se);
            if (ret < 0)
              return ret;

            /* Copy counter data to memory */
            pm4_copy_data_flags_t flags = {
                .bits = {.src_sel = 0,      /* Non-priv registers */
                         .dst_sel = 2,      /* TC_L2 memory */
                         .src_temporal = 3, /* LU cache policy */
                         .dst_temporal = 3, /* LU cache policy */
                         .count_sel = 0,    /* 32-bit data */
                         .wr_confirm = 0}};

            ret = pm4_append_copy_data(
                buffer, flags, reg_info->register_addr_lo,
                reg_info->register_addr_hi, current_addr);
            if (ret < 0)
              return ret;

            current_addr += 8; /* 64-bit counter value */
          }
        }
      }
    } else {
      /* No SE dimension found - global block, read once */
      pm4_copy_data_flags_t flags = {
          .bits = {.src_sel = 0,      /* Non-priv registers */
                   .dst_sel = 2,      /* TC_L2 memory */
                   .src_temporal = 3, /* LU cache policy */
                   .dst_temporal = 3, /* LU cache policy */
                   .count_sel = 0,    /* 32-bit data */
                   .wr_confirm = 0}};

      ret = pm4_append_copy_data(buffer, flags, reg_info->register_addr_lo,
                                 reg_info->register_addr_hi, current_addr);
      if (ret < 0)
        return ret;

      current_addr += 8; /* 64-bit counter value */
    }
  }

  /* 5. Restore broadcast mode */
  ret = generate_grbm_broadcast(buffer, arch);
  if (ret < 0)
    return ret;

  /* 6. Cache coherency flush */
  ret = pm4_append_acquire_mem(buffer, collection->gpu_memory_addr,
                               collection->memory_size,
                               arch->control_regs.gcr_cntl_default);
  if (ret < 0)
    return ret;

  return 0;
}

/* Generate PM4 packet sequence to stop performance counters */
int generate_stop_packet(pm4_buffer_t *buffer, const arch_t *arch) {
  int ret;

  if (!buffer || !arch) {
    return -EINVAL;
  }

  /* 1. GRBM broadcast mode */
  ret = generate_grbm_broadcast(buffer, arch);
  if (ret < 0)
    return ret;

  /* 2. Set perfmon state to stop with sampling enabled */
  ret = generate_perfmon_enable(
      buffer, arch, arch->control_regs.perfmon_states.perfmon_state_stop,
      false);
  if (ret < 0)
    return ret;

  /* 3. CS partial flush */
  ret = generate_cs_partial_flush(buffer, arch);
  if (ret < 0)
    return ret;

  return 0;
}

/* Calculate total memory size needed for counter data collection */
size_t calculate_counter_memory_size(const arch_t *arch,
                                     const counter_collection_t *collection) {
  if (!arch || !collection || !collection->counters) {
    return 0;
  }

  size_t total_size = 0;

  for (size_t i = 0; i < collection->counter_count; i++) {
    counter_info_t *counter = &collection->counters[i];
    block_info_t *block = arch->block_map.blocks[counter->block_id];

    if (!block) {
      continue; /* Skip invalid blocks */
    }

    size_t counter_size = 8; /* Base 64-bit counter size */

    /* Multiply by topology dimensions - use block-specific dimension sizes */
    for (size_t dim_idx = 0; dim_idx < block->dimension_count; dim_idx++) {
      dimension_t *dim = &block->dimensions[dim_idx];
      /* Always use the dimension size from the block, not the arch defaults */
      counter_size *= dim->size;
    }

    total_size += counter_size;
  }

  return total_size;
}

/* Validate counter collection configuration */
int validate_counter_collection(const arch_t *arch,
                                const counter_collection_t *collection) {
  if (!arch || !collection || !collection->counters) {
    return -EINVAL;
  }

  if (collection->counter_count == 0) {
    return -EINVAL;
  }

  /* Check each counter */
  for (size_t i = 0; i < collection->counter_count; i++) {
    counter_info_t *counter = &collection->counters[i];

    /* Validate block ID */
    if (VALIDATE_BLOCK_ID(arch, counter)) {
      return -EINVAL;
    }

    /* Check if block exists in architecture */
    block_info_t *block = arch->block_map.blocks[counter->block_id];
    if (!block) {
      return -ENOENT;
    }

    /* Validate counter index */
    if (counter->counter_index >= block->counter_count) {
      return -EINVAL;
    }

    /* Validate event ID */
    if (counter->event_id > block->event_id_max) {
      return -EINVAL;
    }
  }

  return 0;
}