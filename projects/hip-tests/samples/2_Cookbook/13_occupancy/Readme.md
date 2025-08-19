# occupancy

- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./occupancy
Manual Configuration with block size 32
kernel Execution time =  0.433ms
Theoretical Occupancy is 40%

Automatic Configuation based on hipOccupancyMaxPotentialBlockSize
Suggested blocksize is 1024, Minimum gridsize is 128
kernel Execution time =  0.037ms
Theoretical Occupancy is 80%

Manual Test PASSED!

Automatic Test PASSED!
```