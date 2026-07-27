# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Roofline plot/client values shared between Python and the browser.

Single source of truth for anything the Python figure builder
and the client assets must agree on.
Values needed by the browser are forwarded through
RooflineViewModel.to_json() rather than being re-hardcoded, so a
value only ever changes here.

Pure calc-math constants live with
the ceiling math in utils.roofline_calc instead; this module only holds the
plot/UI contract.
"""

KERNEL_NAME_FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Memory region the roofline opens on when it is present.
DEFAULT_PEAK = "HBM"

# Sentinel value for the "every memory level" dropdown option.
ALL_PEAKS_VALUE = "all"
ALL_PEAKS_LABEL = "All peaks"

# Marker color used when a level or kernel has no assigned color.
FALLBACK_COLOR = "#888888"

# Opacity applied to the non-isolated roof/ceiling traces while isolating.
PLOT_DIM_OPACITY = 0.15

# Log-axis fallback frame when there is no data.
DEFAULT_AXIS_BOUNDS = (0.01, 1000.0, 1.0, 100000.0)

# Initial framing starts on the kernel scatter.
FRAME_PAD = 1.6
FRAME_MIN_DECADES = 2.5
FRAME_ROOF_SEGMENT_DECADES = 2

# Roof sampling is used for hoverability.
ROOF_SAMPLES = 700
# Roofs are extrapolated to very small and very large AI values.
ROOF_EXTRAP_MIN_AI = 1e-150
ROOF_EXTRAP_MAX_AI = 1e150
# Decades beyond the data extremes the roofs stay densely sampled (each side).
ROOF_DENSE_PAD_FACTOR = 1e3

# Per memory-level / compute-roof trace colors for the HTML and CLI backends.
TRACE_COLORS: dict[str, dict[str, str]] = {
    "l0": {"html": "#F0E442", "cli": "brown+"},
    "l1": {"html": "#0072B2", "cli": "red+"},
    "l2": {"html": "#009E73", "cli": "green+"},
    "hbm": {"html": "#D55E00", "cli": "blue+"},
    "lds": {"html": "#E69F00", "cli": "orange+"},
    "valu": {"html": "#CC79A7", "cli": "white"},
    "matrix_ops": {"html": "#56B4E9", "cli": "magenta+"},
}
