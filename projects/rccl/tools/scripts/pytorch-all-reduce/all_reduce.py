import os
import torch
import torch.distributed as dist
import time
import argparse
import statistics

def init_process(rank, size, fn, backend='nccl'):
    """ Init the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)

    dist.init_process_group(backend, rank=rank, world_size=size)
    return fn(rank, size)

def get_algo_type(data_size):
    if data_size <= 256 * 1024:
        return "one-shot allreduce"
    elif data_size <= 10 * 1024 * 1024:
        return "two-shot allreduce"
    else:
        return "nccl"

def benchmark_all_reduce(rank, size, sequence_lengths, dim, all_reduce_algos, tracing):
    """ Benchmark all-reduce operation - 4 different datasizes will be benched per run """
    torch.cuda.set_device(rank)

    n_runs = 1000

    results = []

    # All-reduce sizes for gpt fast
    algo_shapes = {
        1: (lambda seq_len: (seq_len, dim)),
        2: (lambda seq_len: (seq_len, 2, dim)),
        3: (lambda _: (1, dim)),
        4: (lambda _: (1, 2, dim))
    }

    for seq_len in sequence_lengths:
        for algo in all_reduce_algos:
            shape = algo_shapes[algo](seq_len)
            main_times = []
            tensor = torch.randn(*shape, device='cuda').to(torch.bfloat16)

            # Warm-up - before result collection 
            for _ in range(5):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                dist.barrier()

            # Benchmark - result collection and timers disabled if --tracing applied 
            for _ in range(n_runs):
                if not tracing: 
                    start = time.time()

                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                dist.barrier()
                
                if not tracing:
                    end = time.time()
                    main_time = (end - start) * 1e6  # Convert to microseconds
                    main_times.append(main_time)

            if rank == 0 and not tracing:
                mean_time = statistics.mean(main_times)
                median_time = statistics.median(main_times)
                max_time = max(main_times)
                min_time = min(main_times)
                std_time = statistics.stdev(main_times)
                data_size_bytes = torch.tensor(shape).prod().item() * 2  # * 2 as bfloat16 takes 2 bytes
                data_size_mb = data_size_bytes / (1024 ** 2)
                algo_type = get_algo_type(data_size_bytes)
                results.append([f"all_reduce_{algo}", seq_len, data_size_mb, algo_type, mean_time, median_time, max_time, min_time, std_time])

    return results if rank == 0 and not tracing else None

def main():
    parser = argparse.ArgumentParser(description='PyTorch All-Reduce Benchmark')
    parser.add_argument('--sequence_lengths', type=int, nargs='+', default=[50, 64, 128, 256, 512, 1024, 2048, 4096],
                        help='List of sequence lengths to benchmark')
    parser.add_argument('--dim', type=int, default=6144, help='Dimension for tensor shapes')
    parser.add_argument('--all_reduce', type=int, nargs='+', choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
                        help='List of all-reduce algorithms to run (1, 2, 3, 4)')
    parser.add_argument('--tracing', action='store_true', help='Enable tracing mode (skip CPU timers and output table)')
    args = parser.parse_args()

    sequence_lengths = args.sequence_lengths
    dim = args.dim
    all_reduce_algos = args.all_reduce
    tracing = args.tracing

    size = int(os.environ['WORLD_SIZE'])  # number of processes (GPUs)
    rank = int(os.environ['RANK'])
    results = init_process(rank, size, fn=lambda rank, size: benchmark_all_reduce(rank, size, sequence_lengths, dim, all_reduce_algos, tracing), backend='nccl')

    if rank == 0 and not tracing:
        header = ["algo", "sequence_length", "data_size (MB)", "algo type", "mean (us)", "median (us)", "max (us)", "min (us)", "std (us)"]
        print(",".join(header))
        for result in results:
            formatted_result = [f"{item:.2f}" if isinstance(item, float) else str(item) for item in result]
            print(",".join(formatted_result))

if __name__ == "__main__":
    main()
