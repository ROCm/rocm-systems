/*
 * pmu_dimension.h - Dimension-aware Performance Monitoring for AMD GPU PMU
 *
 * This header defines structures and helpers for dimension-specific performance
 * counter monitoring. It enables users to target specific hardware instances
 * (XCC, SE, SA, WGP, CU) when creating perf events.
 *
 * Example usage:
 *   perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1/ command
 *   perf stat -e amdgpu_pmu/sq_waves,config1=0x00000100/ command
 */

#ifndef _PMU_DIMENSION_H
#define _PMU_DIMENSION_H

#include <linux/types.h>

/*
 * config1 Bit Field Layout for Dimension Encoding
 * ================================================
 *
 * config1 is a 64-bit field that encodes hardware dimension coordinates:
 *
 *   Bits  0-7   : XCC index (0-255)
 *   Bits  8-15  : SE index (0-255)
 *   Bits 16-23  : SA index (0-255)
 *   Bits 24-31  : WGP index (0-255)
 *   Bits 32-39  : CU index (0-255)
 *   Bit  40     : Aggregate flag (1 = aggregate across dimensions)
 *   Bits 41-63  : Reserved (must be zero)
 *
 * This encoding allows perf to specify dimensions either via named parameters
 * (e.g., "se=2,sa=1") or raw config1 values (e.g., "config1=0x00010200").
 *
 * Default Behavior:
 *   When dimensions are not specified (config1=0), the driver operates in
 *   broadcast mode: all hardware instances are read and the results are
 *   aggregated (summed). When specific dimensions are provided (e.g., se=2),
 *   unspecified sub-dimensions default to 0 (e.g., se=2 means se=2,sa=0,wgp=0).
 */

/* Bit field layout constants */
#define PMU_DIM_XCC_SHIFT       0
#define PMU_DIM_XCC_MASK        0xFF
#define PMU_DIM_SE_SHIFT        8
#define PMU_DIM_SE_MASK         0xFF
#define PMU_DIM_SA_SHIFT        16
#define PMU_DIM_SA_MASK         0xFF
#define PMU_DIM_WGP_SHIFT       24
#define PMU_DIM_WGP_MASK        0xFF
#define PMU_DIM_CU_SHIFT        32
#define PMU_DIM_CU_MASK         0xFF
#define PMU_DIM_AGGREGATE_SHIFT 40

/**
 * struct pmu_dimension_coords - Hardware dimension coordinates
 *
 * Represents the specific hardware instance to monitor. All dimension
 * indices are zero-based.
 *
 * @xcc: XCC (Infinity Fabric block) index
 * @se: Shader Engine index
 * @sa: Shader Array index
 * @wgp: Work Group Processor index
 * @cu: Compute Unit index
 * @valid: True if coordinates were explicitly specified by user
 * @aggregate: True to aggregate counts across all matching instances
 */
struct pmu_dimension_coords {
	u8 xcc;
	u8 se;
	u8 sa;
	u8 wgp;
	u8 cu;
	bool valid;
	bool aggregate;
};

/**
 * struct pmu_dimension_limits - Maximum values for each dimension
 *
 * Defines the hardware topology limits for the current GPU. These are
 * populated from the architecture structure during module initialization.
 *
 * @max_xcc: Maximum XCC index (num_xcc - 1)
 * @max_se: Maximum SE index (num_se - 1)
 * @max_sa: Maximum SA index (num_sa - 1)
 * @max_wgp: Maximum WGP index (num_wgp_per_sa - 1)
 * @max_cu: Maximum CU index (num_cu - 1)
 */
struct pmu_dimension_limits {
	u8 max_xcc;
	u8 max_se;
	u8 max_sa;
	u8 max_wgp;
	u8 max_cu;
};

/**
 * pmu_extract_dimensions - Extract dimension coordinates from config1
 * @config1: The perf event config1 field
 * @dims: Output structure to populate
 *
 * Extracts dimension coordinates from the config1 bit field and populates
 * the provided dimension structure. Sets the 'valid' flag if any non-zero
 * dimension or flag is present.
 *
 * This function is called during event initialization to decode user-specified
 * dimensions from the perf event attributes.
 *
 * Default behavior: Unspecified dimensions default to 0. For example, if the
 * user specifies only se=2, then sa, wgp, and cu will be 0 (i.e., se=2,sa=0,
 * wgp=0,cu=0). If config1 is 0 (no dimensions specified), the driver uses
 * broadcast mode and aggregates across all instances.
 */
static inline void pmu_extract_dimensions(u64 config1,
					  struct pmu_dimension_coords *dims)
{
	dims->xcc = (config1 >> PMU_DIM_XCC_SHIFT) & PMU_DIM_XCC_MASK;
	dims->se = (config1 >> PMU_DIM_SE_SHIFT) & PMU_DIM_SE_MASK;
	dims->sa = (config1 >> PMU_DIM_SA_SHIFT) & PMU_DIM_SA_MASK;
	dims->wgp = (config1 >> PMU_DIM_WGP_SHIFT) & PMU_DIM_WGP_MASK;
	dims->cu = (config1 >> PMU_DIM_CU_SHIFT) & PMU_DIM_CU_MASK;
	dims->aggregate = (config1 >> PMU_DIM_AGGREGATE_SHIFT) & 1;

	/* Mark as valid if any dimension or flag is non-zero */
	dims->valid = (config1 != 0);
}

/**
 * pmu_validate_dimensions - Validate dimension coordinates against limits
 * @dims: Dimension coordinates to validate
 * @limits: Hardware topology limits
 *
 * Returns: true if coordinates are within hardware limits, false otherwise
 *
 * Checks that all specified dimension indices are within the valid range
 * for the current GPU architecture. Invalid coordinates indicate a user
 * error and should result in event creation failure.
 *
 * Note: This function assumes that if dims->valid is false, no validation
 * is needed (aggregate/broadcast mode).
 */
static inline bool pmu_validate_dimensions(const struct pmu_dimension_coords *dims,
					   const struct pmu_dimension_limits *limits)
{
	/* If dimensions not specified, nothing to validate */
	if (!dims->valid)
		return true;

	/* Check each dimension against limits */
	if (dims->xcc > limits->max_xcc)
		return false;
	if (dims->se > limits->max_se)
		return false;
	if (dims->sa > limits->max_sa)
		return false;
	if (dims->wgp > limits->max_wgp)
		return false;
	if (dims->cu > limits->max_cu)
		return false;

	return true;
}

/**
 * pmu_encode_dimensions - Encode dimension coordinates into config1 format
 * @dims: Dimension coordinates to encode
 *
 * Returns: config1 value with encoded dimensions
 *
 * Converts a dimension coordinate structure back into the config1 bit field
 * format. This is primarily used for debugging and testing.
 */
static inline u64 pmu_encode_dimensions(const struct pmu_dimension_coords *dims)
{
	u64 config1 = 0;

	config1 |= ((u64)dims->xcc & PMU_DIM_XCC_MASK) << PMU_DIM_XCC_SHIFT;
	config1 |= ((u64)dims->se & PMU_DIM_SE_MASK) << PMU_DIM_SE_SHIFT;
	config1 |= ((u64)dims->sa & PMU_DIM_SA_MASK) << PMU_DIM_SA_SHIFT;
	config1 |= ((u64)dims->wgp & PMU_DIM_WGP_MASK) << PMU_DIM_WGP_SHIFT;
	config1 |= ((u64)dims->cu & PMU_DIM_CU_MASK) << PMU_DIM_CU_SHIFT;
	config1 |= ((u64)dims->aggregate & 1) << PMU_DIM_AGGREGATE_SHIFT;

	return config1;
}

#endif /* _PMU_DIMENSION_H */
