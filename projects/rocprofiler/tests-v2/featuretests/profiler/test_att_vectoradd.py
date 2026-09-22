import numpy as np
import pandas

MAX_CU = 16
MAX_WAVE_SIZE = 64
MAXIMUM_ATT_HITS = 256*128//MAX_WAVE_SIZE//MAX_CU

kernel_name = "vectoradd_att"
csv_filename = "vadd_" + kernel_name + "_v0.csv"
output_folder = "/tmp/tests-v2/att"


def test_hitcount(csv):
    hits = {m: True for m in csv['Hitcount'] if m != 0}
    print('hits', hits)
    assert(len(hits) > 0)
    assert(np.max([k for k in hits.keys()]) <= MAXIMUM_ATT_HITS)


def test_addr(csv):
    addrs = np.array([int(addr, 16) for addr in csv['Addr'] if addr != '0x0'])
    print('addrs', addrs)
    assert(addrs.max() - addrs.min() > 32)       # 32 bytes is a safe minimum value
    assert(addrs.max() - addrs.min() < 2**24)    # Kernels are not anywhere near that large


def test_memory_list(csv):
    inst_list = ' '.join(csv['Instruction'])
    assert('vectoradd_' in inst_list)
    assert('s_load_' in inst_list)
    assert('_store_' in inst_list)
    assert('s_waitcnt' in inst_list)
    assert('v_add' in inst_list)
    assert('global_load' in inst_list or 'buffer_load' in inst_list or 'flat_load' in inst_list)


def test_mean_cycles(csv):
    cycles = np.array([c/float(h) for c, h in zip(csv['Cycles'], csv['Hitcount']) if c != 0])
    print('cycles', cycles)
    assert(cycles.min() < 5)        # Waves should have some instructions with very few cycles
    assert(cycles.max() > 100)      # s_waitcnt should have a large cost
    assert(cycles.mean() > 1)       # Minimum cost per inst is 1
    assert(np.median(cycles) <= 16) # Majority of instructions are not that expensive

    maxv = int(4*cycles.max()+5)//4
    histogram = np.histogram(cycles, range=[0,maxv], bins=max(maxv//8, 1))[0]
    assert(histogram[0] == np.max(histogram))  # 1~8 cycles should be most common cost


def test_memory_cycles(csv):
    is_memory_op = lambda s: ('waitcnt' in s) or ('_load_' in s) or ('_store_' in s)

    max_cycles = np.max(csv['Cycles'])
    most_exp_inst = [f for f in csv[csv['Cycles'] == max_cycles]['Instruction']][0]
    print('most_exp_inst', most_exp_inst)
    assert(is_memory_op(most_exp_inst)) # Memory ops should be the most expensive insts

    memory_ops = [c for s, c in zip(csv['Instruction'],csv['Cycles']) if is_memory_op(s)]
    print('memory_ops', memory_ops) # Memory ops should be more than half the total cycles
    assert(np.sum(memory_ops) > np.sum(csv['Cycles'])*0.5)


if __name__ == "__main__":
    csv = pandas.read_csv(f"{output_folder}/{csv_filename}")

    test_hitcount(csv)
    test_addr(csv)
    test_memory_list(csv)
    test_mean_cycles(csv)
    test_memory_cycles(csv)

    print("Test Passed: All ATT correctness tests passed.")
