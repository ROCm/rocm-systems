#run ctest
#cd /home/rocm/AnujWork/rocpdsna/rocm-systems/projects/rocpdsna/build 
#ctest --test-dir tests/



#cd /home/rocm/AnujWork/rocpdsna/rocm-systems/projects/rocpdsna/build_v4/bin
#clean preveous 
# rm -f *.db
# rm -f V4_benchmark_results_50.csv
# ./rocpdsna_benchmarks --benchmark_repetitions=50 --benchmark_format=csv  --benchmark_out=V4_benchmark_results_50.csv --benchmark_out_format=csv

#sleep 120
cd /home/rocm/AnujWork/rocpdsna/rocm-systems/projects/rocpdsna/build_v/bin
export LD_LIBRARY_PATH=/home/rocm/AnujWork/rocpdsna/rocm-systems/projects/rocprofiler-sdk/build/lib:$LD_LIBRARY_PATH
rm -f *.db
rm -f V3_benchmark_results_50.csv
./rocpdsna_benchmarks --benchmark_repetitions=50 --benchmark_format=csv  --benchmark_out=V3_benchmark_results_50.csv  --benchmark_out_format=csv