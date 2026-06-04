#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#ifdef RJ_AFL_PERSISTENT_MODE
#include <unistd.h>

__AFL_FUZZ_INIT();
#endif

namespace {

void say_hello(std::string_view input) {
  while (!input.empty() && (input.back() == '\n' || input.back() == '\r')) {
    input.remove_suffix(1);
  }

  if (input == "hello") {
    std::cout << "Hello, AFL!\n";
  } else if (input.substr(0, 4) == "rocm") {
    std::cout << "Hello, ROCm!\n";
  } else if (!input.empty()) {
    std::cout << "Hello, fuzzed input!\n";
  }
}

} // namespace

int main() {
#ifdef RJ_AFL_PERSISTENT_MODE
  __AFL_INIT();

  unsigned char *buffer = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(10000)) {
    const int length = __AFL_FUZZ_TESTCASE_LEN;
    say_hello(std::string_view(reinterpret_cast<const char *>(buffer), length));
  }
#else
  std::string input;
  std::getline(std::cin, input);
  say_hello(input);
#endif

  return EXIT_SUCCESS;
}
