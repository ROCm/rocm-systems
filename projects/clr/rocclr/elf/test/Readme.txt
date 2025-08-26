1. To build release version
In test folder,
mkdir release (if release doesn't exist)
cd release
cmake ..
make


2. To build debug version
In test folder,
mkdir debug (if debug doesn't exist)
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

3. Run test
rm -f *.bin
./elf_test

To get debug log,
AMD_LOG_LEVEL=5 ./elf_test
