#include <unistd.h>
#include <iostream>
#include "hsatimer.hpp"
#include "rocm_async.hpp"

using namespace std;

int main(int argc, char** argv) {

  // Create the Bandwidth test object
  RocmAsync bw_test(argc, argv);

  // Initialize the Bandwidth test object
  bw_test.SetUp();

  // Run the Bandwidth tests requested by user
  bw_test.Run();

  // Display the time taken by various tests
  bw_test.Display();

  // Release the Bandwidth test object resources
  bw_test.Close();
  return 0;
}
