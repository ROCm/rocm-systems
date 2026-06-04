#include <cstdlib>
#include <iostream>
#include <string>

int main() {
  std::string s;
  std::getline(std::cin, s);

  if (s == "hello") {
    std::cout << "Hello, AFL!\n";
  } else if (s.rfind("rocm", 0) == 0) {
    std::cout << "Hello, ROCm!\n";
  } else if (!s.empty()) {
    std::cout << "Hello, fuzzed input!\n";
  }

  return EXIT_SUCCESS;
}
