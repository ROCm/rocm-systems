rm *.code *.o *.a bin

hipcc --genco MultiplyKernel.cpp -o MultiplyKernel.code

# common
hipcc -c Common.cpp -o Common.o
ar rcs libCommon.a Common.o

# Multiplication
hipcc -c Multiplication.cpp -L. -lCommon -o Multiplication.o
ar rcs libMul.a Multiplication.o

# Addition
hipcc -c Addition.cpp -L. -lCommon -o Addition.o
ar rcs libAdd.a Addition.o

# Compile App
hipcc Application.cpp -L. -lMul -lAdd -lCommon -o bin

# Run
./bin
