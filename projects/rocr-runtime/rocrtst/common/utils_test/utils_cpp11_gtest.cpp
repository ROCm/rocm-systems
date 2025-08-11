#include<iostream>
#include<thread>
#include"gtest/gtest.h"

using std::cout;
using std::endl;

// @Brief: this function is defined to be executed for thread #1
static void ThreadEntry1() {
  cout << "The first thread is launched!" << endl;
  return;
}
// @Brief: this function is defined to be executed for thread #2
static void ThreadEntry2() {
  cout << "The second thread is launched!" << endl;
  return;
}

// @Brief: google test case added for basic C++11 thread feature.
// Here, in main function, it will create two threas objects, then,
// check if each thread are joinable, if so, main thread wait until
// the spawned threads finish.
TEST(rocrtstCpp11Feature, BasicThread) {
  // Define two threads object;
  std::thread thread1;
  std::thread thread2;

  // At this point, it should be non-joinable
  ASSERT_EQ(false, thread1.joinable());
  ASSERT_EQ(false, thread2.joinable());

  // Assign execution codes to threads;
  thread1 = std::thread(ThreadEntry1);
  thread2 = std::thread(ThreadEntry2);

  // Now, the two threads should be joinable
  ASSERT_EQ(true, thread1.joinable());
  ASSERT_EQ(true, thread2.joinable());

  // Join the two threads until they finish
  thread1.join();
  thread2.join();

  // When execution flow reaches here, it succeed.
  cout << "Done!" << endl;
}
