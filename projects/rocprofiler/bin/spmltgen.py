import sys
import enum

#source file handling: from last command line argument or default
argc = len(sys.argv)
if argc == 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage:")
    print("python3 spmltgen.py <gfx> <input counter event space delimited list: SQ_WAVES ... [<block>_<event>]>")
    print("python3 spmltgen.py <gfx> -f counter_events_file")
    print("  <gfx> GPU gfx id, i.e. gfx908")
    print("  <block> SPM block name, one of CPC, CPF, GDS, SPI, SQ, SX, TA, TCA, TCC, TCP, TD")
    print("  <event> SPM block event names specified by <block> above")
    print("  counter_events_file lists one event counter <block>_<event> per line"
    print("  spmltgen.py print out the layout of counter events in SPM sample data buffer")
    print("  Output layout lines format:")
    print("  <number of 256bit aligned line instances>: <space delimeted timestamp/event/gap list>")
    print("    T64 - for 64bit timestamp")
    print("    X<N> - for Nbit gap")
    sys.exit(0)

# set the number of shader engines
se_no = 4
if sys.argv[1] == "gfx908" or sys.argv[1] == "gfx90a" or sys.argv[1] == "gfx1030":
    se_no = 8
else:
    print("Unknown gfx chip: " + sys.argv[1])
    sys.exit(0)

# test if a block is a global block
def is_global_block(block_name):
    if block_name == "CPC" or block_name == "CPF" or block_name == "GDS" or block_name == "TCC" or block_name == "TCA":
        return True
    else:
        return False

# extract counter events from either the command line or the counter_events_file
counter_args = []
if sys.argv[2] == "-f":
    if argc < 4:
        print("Missing counter_event_file")
        sys.exit(0)
    file = open(sys.argv[3], 'r')
    for line in file:
        counter_args.append(line.strip())
    file.close()
else:
    for i in range(argc):
        if i > 1:
            counter_args.append(sys.argv[i])

# build counter events
counters = []
for event in counter_args:
    block_name = event.partition('_')[0]
    name = event.partition(':')[0]
    id = event.partition(':')[2]
    counter_des = []
    counter_des.append(block_name)
    counter_des.append(name)
    counter_des.append(id)
    counters.append(counter_des)

even_counters = []
odd_counters = []

# SPM block id enum
class block_idx(enum.IntEnum):
    CPC = 0
    CPF = 1
    GDS = 2
    SPI = 3
    SQ  = 4
    SX  = 5
    TA  = 6
    TCA = 7
    TCC = 8
    TCP = 9
    TD  = 10

# Return SPM block id enum from its block name
def get_block_idx(block_name):
    if block_name == "CPC":
        return block_idx.CPC
    elif block_name == "CPF":
        return block_idx.CPF
    elif block_name == "GDS":
        return block_idx.GDS
    elif block_name == "SPI":
        return block_idx.SPI
    elif block_name == "SQ":
        return block_idx.SQ
    elif block_name == "SX":
        return block_idx.SX
    elif block_name == "TA":
        return block_idx.TA
    elif block_name == "TCA":
        return block_idx.TCA
    elif block_name == "TCC":
        return block_idx.TCC
    elif block_name == "TCP":
        return block_idx.TCP
    elif block_name == "TD":
        return block_idx.TD
    else:
        print("Unknown SPM block name: " + block_name + " encountered")
        print("Exiting ...")
        sys.exit(0)

# is SQ SPM counters treated as 32-bit mode
sq_32b_mode = True

# Odd and even counter events (dictionary value list) according to each SPM block id (dictionary key)
odd = {}
even = {}

# Check if a key is already in the dictionary
def check_key(dict, key):
    if key in dict.keys():
        return True
    return False

# Assign counter events of a block (id used as key) to odd and even events (event list as value) of its own
for counter in counters:
    key = get_block_idx(counter[0])
    is_odd_inserted = check_key(odd, key)
    is_even_inserted = check_key(even, key)
    if counter[0] == "SQ" and sq_32b_mode:
        if not is_even_inserted:
            evens = []
            evens.append(counter)
            even[key] = evens
        else:
            evens = even[key]
            evens.append(counter)
            even[key] = evens
        if not is_odd_inserted:
            odds = []
            odds.append(counter)
            odd[key] = odds
        else:
            odds = odd[key]
            odds.append(counter)
            odd[key] = odds
    else:
        if not is_even_inserted:
            evens = []
            evens.append(counter)
            even[key] = evens
        elif not is_odd_inserted:
            odds = []
            odds.append(counter)
            odd[key] = odds
        elif len(even[key]) == len(odd[key]):
            # push to evens
            evens = even[key]
            evens.append(counter)
            even[key] = evens
        else:
            # push to odds
            odds = odd[key]
            odds.append(counter)
            odd[key] = odds

# Total global and shader engine even and odd counter events
cntr_array_even = [["T64"], []]
cntr_array_odd = [[], []]

# Assign all odd and even counter events from each block to the total global and shader engine odd and even counter events
for s in block_idx:
    if check_key(even, s):
        even_blk_counters = even[s]
        even_blk_counter = even_blk_counters[0]
        if is_global_block(even_blk_counter[0]):
            for even_blk_counter in even_blk_counters:
                cntr_array_even[0].append(even_blk_counter[1])
        else:
            for even_blk_counter in even_blk_counters:
                cntr_array_even[1].append(even_blk_counter[1])

    if check_key(odd, s):
        odd_blk_counters = odd[s]
        odd_blk_counter = odd_blk_counters[0]
        if is_global_block(odd_blk_counter[0]):
            for odd_blk_counter in odd_blk_counters:
                cntr_array_odd[0].append(odd_blk_counter[1])
        else:
            for odd_blk_counter in odd_blk_counters:
                cntr_array_odd[1].append(odd_blk_counter[1])

# Accumulated even and odd counter events (max 16 events per even or odd) for print set to empty
even_layout = []
odd_layout = []

# Print 16-event even and odd segments on a single line
def print_layout(odd_layout, even_layout):
    for i in even_layout:
        print(i, end = " ")
    for i in odd_layout:
        print(i, end = " ")
    print()

# Function to print all global counter events per 16-event even and odd segment
def print_global_layout(odd_counters, even_counters):
    global even_layout
    global odd_layout
    # segments of even event counters
    segs = (len(even_counters) + 3) >> 4
    if (len(even_counters) % 16) != 0:
        segs += 1
    idx_even = 0
    idx_odd = 0
    even_padding = "X0"
    odd_padding = "X0"
    for i in range(segs * 16):
        # fill in 16 even global counters as they exist
        if i >= 13 and i <= 15:
            pass
        elif i < len(even_counters):
            even_layout.append(even_counters[idx_even])
            idx_even += 1
        else:
            # padding with X
            padding = int(even_padding[1:])
            padding += 16
            even_padding = "X" + str(padding)
        # fill in corresponding odd 16 global counters as they exist
        if i < len(odd_counters):
            odd_layout.append(odd_counters[idx_odd])
            idx_odd += 1
        else:
            # padding with X
            padding = int(odd_padding[1:])
            padding += 16
            odd_padding = "X" + str(padding)
        if i % 16 == 15:
            # after accumulated one segment of both even and odd events print their layout
            if even_padding != "X0":
                even_layout.append(even_padding)
                even_padding = "X0"
            if odd_padding != "X0":
                odd_layout.append(odd_padding)
                odd_padding = "X0"
            print("1: ", end = '')
            print_layout(odd_layout, even_layout)
            even_layout = []
            odd_layout = []

# Accumulated even and odd counter events (max 16 events per even or odd) for print set to empty
even_layout = []
odd_layout = []

# Function to print all shader engine counter events per 16-event even and odd segment
def print_se_layout(se_no, odd_counters, even_counters):
    global even_layout
    global odd_layout
    print(str(se_no) + ": ", end = '')
    # segments of even event counters
    segs = len(even_counters) >> 4
    if (len(even_counters) % 16) != 0:
        segs += 1
    even_padding = "X0"
    odd_padding = "X0"
    # process each event in all segments
    for i in range(segs * 16):
        # fill in 16 even global counters as they exist
        if i < len(even_counters):
            even_layout.append(even_counters[i])
        else:
            # padding with X
            padding = int(even_padding[1:])
            padding += 16
            even_padding = "X" + str(padding)
        # fill in corresponding odd 16 global counters as they exist
        if i < len(odd_counters):
            odd_layout.append(odd_counters[i])
        else:
            # padding with X
            padding = int(odd_padding[1:])
            padding += 16
            odd_padding = "X" + str(padding)
        if i % 16 == 15:
            # after accumulated one segment of both even and odd events print their layout
            if even_padding != "X0":
                even_layout.append(even_padding)
                even_padding = "X0"
            if odd_padding != "X0":
                odd_layout.append(odd_padding)
                odd_padding = "X0"
            print_layout(odd_layout, even_layout)
            even_layout = []
            odd_layout = []

# Print alll even and odd events
print_global_layout(cntr_array_odd[0], cntr_array_even[0])
print_se_layout(se_no, cntr_array_odd[1], cntr_array_even[1])
