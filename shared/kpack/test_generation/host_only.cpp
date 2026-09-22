// Simple host-only C++ code with no GPU device code
// Used for negative testing of bundled binary detection

#include <iostream>

extern "C" int add(int a, int b) { return a + b; }

extern "C" void print_hello() {
  std::cout << "Hello from host-only code" << std::endl;
}

#ifdef BUILD_EXECUTABLE
int main() {
  std::cout << "Host-only executable: " << add(5, 3) << std::endl;
  return 0;
}
#endif
