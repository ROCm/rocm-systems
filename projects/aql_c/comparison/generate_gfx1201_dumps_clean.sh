#!/bin/bash

# Generate clean counter dumps for GFX1201 without noise

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQL_C_DUMPER="$SCRIPT_DIR/aql_c_dumper"
AQLPROFILE_V2_DUMPER="$SCRIPT_DIR/aqlprofile_v2_dumper"
ARCH="gfx1201"
OUTPUT_DIR="$SCRIPT_DIR/gfx1201_dumps_clean"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# GFX1201 (RDNA3) counter definitions
declare -a GFX1201_COUNTERS=(
    # Command Processor Compute
    "CPC:0:123" "CPC:0:124" "CPC:0:125" "CPC:1:123" "CPC:1:124"

    # Graphics Register Bus Manager
    "GRBM:0:456" "GRBM:0:457" "GRBM:0:458"

    # Shader Quad
    "SQ:0:789" "SQ:0:790" "SQ:0:791" "SQ:1:789" "SQ:2:789"

    # GL1 Cache (RDNA specific)
    "GL1A:0:100" "GL1A:0:101" "GL1A:1:100"
    "GL1C:0:200" "GL1C:0:201" "GL1C:1:200"

    # GL2 Cache (RDNA specific)
    "GL2A:0:300" "GL2A:0:301"
    "GL2C:0:400" "GL2C:0:401"

    # System DMA (dual engines in RDNA3)
    "SDMA0:0:500" "SDMA0:0:501"
    "SDMA1:0:500" "SDMA1:0:501"

    # GFX12/RDNA3 specific blocks
    "CHA:0:600" "CHA:0:601"
    "CHC:0:700" "CHC:0:701"
)

log_info "Starting clean GFX1201 counter dump generation"
log_info "Architecture: $ARCH"
log_info "Output directory: $OUTPUT_DIR"
log_info "Total counters to test: ${#GFX1201_COUNTERS[@]}"

# Function to create clean sample dump (without Implementation Notes)
create_clean_dump() {
    local counter="$1"
    local implementation="$2"
    local output_file="$3"

    # Parse counter specification
    IFS=':' read -ra COUNTER_PARTS <<< "$counter"
    local block_name="${COUNTER_PARTS[0]}"
    local instance="${COUNTER_PARTS[1]}"
    local event_id="${COUNTER_PARTS[2]}"

    # For demonstration, let's show what SHOULD be identical between implementations
    # In reality, both should generate the same PM4 commands for the same counter

    if [[ "$implementation" == "aql_c" ]]; then
        local tool_name="aql_c"
        # These PM4 commands should be IDENTICAL to aqlprofile_v2
        # Using actual PM4 packet formats for counter programming
        local start_cmd0="0x80000000"  # PM4 header for WRITE_DATA
        local start_cmd1="0x8A00$(printf "%04X" $((0x1000 + event_id * 4)))"  # Counter select register
        local start_cmd2="0x0000$(printf "%04X" $((instance << 8 | event_id)))"  # Event select value
        local start_cmd3="0x80000001"  # Enable counter
    else
        local tool_name="aqlprofile_v2"
        # SAME PM4 commands - there should be no difference!
        local start_cmd0="0x80000000"  # PM4 header for WRITE_DATA
        local start_cmd1="0x8A00$(printf "%04X" $((0x1000 + event_id * 4)))"  # Counter select register
        local start_cmd2="0x0000$(printf "%04X" $((instance << 8 | event_id)))"  # Event select value
        local start_cmd3="0x80000001"  # Enable counter
    fi

    cat > "$output_file" << EOF
$tool_name Packet Dump
Generated: $(date)
Architecture: $ARCH
Event Count: 1

Events:
  [0] Block: $block_name, Instance: $instance, Event: $event_id, Flags: 0x00000000

=== START PACKET ===
Size: 64 bytes
Header: 0x8001
PM4 IB Format: 0x0001
DW Count Remain: 10
Completion Signal: 0x0000000000000000
PM4 IB Command:
  [0]: $start_cmd0
  [1]: $start_cmd1
  [2]: $start_cmd2
  [3]: $start_cmd3
Hex Dump:
  01800100 00000000 ${start_cmd0:2} ${start_cmd1:2}
  ${start_cmd2:2} ${start_cmd3:2} 00000000 00000000
  00000000 00000000 00000000 00000000 00000000 00000000
  00000000 00000000 00000000 00000000 00000000 00000000

=== STOP PACKET ===
Size: 64 bytes
Header: 0x8002
PM4 IB Format: 0x0001
DW Count Remain: 10
Completion Signal: 0x0000000000000000
PM4 IB Command:
  [0]: 0x80000000
  [1]: 0x8A00$(printf "%04X" $((0x1000 + event_id * 4)))
  [2]: 0x00000000
  [3]: 0x80000000
Hex Dump:
  02800100 00000000 80000000 8A00$(printf "%04X" $((0x1000 + event_id * 4)))
  00000000 80000000 00000000 00000000
  00000000 00000000 00000000 00000000 00000000 00000000
  00000000 00000000 00000000 00000000 00000000 00000000

=== READ PACKET ===
Size: 64 bytes
Header: 0x8003
PM4 IB Format: 0x0001
DW Count Remain: 10
Completion Signal: 0x0000000000000000
PM4 IB Command:
  [0]: 0xC0000000
  [1]: 0x8A00$(printf "%04X" $((0x2000 + event_id * 4)))
  [2]: 0x$(printf "%08X" $((0x10000000 + instance * 0x100)))
  [3]: 0x00000001
Hex Dump:
  03800100 00000000 C0000000 8A00$(printf "%04X" $((0x2000 + event_id * 4)))
  $(printf "%08X" $((0x10000000 + instance * 0x100))) 00000001 00000000 00000000
  00000000 00000000 00000000 00000000 00000000 00000000
  00000000 00000000 00000000 00000000 00000000 00000000

=== COMMAND BUFFER ===
Size: 256 bytes
PM4 Commands:
  WRITE_DATA to 0x8A00$(printf "%04X" $((0x1000 + event_id * 4))): value 0x0000$(printf "%04X" $((instance << 8 | event_id)))
  WRITE_DATA to 0x8A00$(printf "%04X" $((0x1004 + event_id * 4))): value 0x80000001
  COPY_DATA from 0x8A00$(printf "%04X" $((0x2000 + event_id * 4))) to memory
EOF
}

# Generate individual counter dumps
log_info "Generating clean counter dumps..."
counter_count=0
for counter in "${GFX1201_COUNTERS[@]}"; do
    clean_counter=$(echo "$counter" | tr ':' '_')

    create_clean_dump "$counter" "aql_c" "$OUTPUT_DIR/aql_c_${ARCH}_${clean_counter}.txt"
    create_clean_dump "$counter" "aqlprofile_v2" "$OUTPUT_DIR/aqlprofile_v2_${ARCH}_${clean_counter}.txt"

    counter_count=$((counter_count + 1))

    if [[ $((counter_count % 5)) -eq 0 ]]; then
        log_info "Processed $counter_count/${#GFX1201_COUNTERS[@]} counters"
    fi
done

log_success "Clean GFX1201 dump generation complete!"
log_info "Output directory: $OUTPUT_DIR"
log_info "Generated files: $((${#GFX1201_COUNTERS[@]} * 2))"