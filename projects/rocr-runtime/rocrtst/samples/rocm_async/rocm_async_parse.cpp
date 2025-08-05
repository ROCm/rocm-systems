#include "common.hpp"
#include "rocm_async.hpp"

#include <algorithm>
#include <sstream>
#include <unistd.h>

// Parse option value string. The string has one more decimal
// values separated by comma - "3,6,9,12,15".
static bool ParseOptionValue(char* value, vector<uint32_t>&value_list) {
 
  // Capture the option value string
  std::stringstream stream;
  stream << value;
  
  uint32_t token = 0x11231926;
  do {
    
    // Read the option value
    stream >> token;

    // Update output list with values
    value_list.push_back(token);

    // Ignore the delimiter
    if((stream.eof()) ||
       (stream.peek() == ',')) {
      stream.ignore();
    } else {
      return false;
    }

  } while (!stream.eof());

  return true;
}

void RocmAsync::ParseArguments() {

  bool print_help = false;
  bool copy_all_bi = false;
  bool copy_all_uni = false;
  bool print_topology = false;

  // This will suppress prints from getopt implementation
  // In case of error, it will return the character '?' as
  // return value.
  opterr = 0;
  
  int opt;
  bool status;
  while ((opt = getopt(usr_argc_, usr_argv_, "hvtaAb:s:d:r:w:m:")) != -1) {
    switch (opt) {

      // Print help screen
      case 'h':
        print_help = true;
        break;

      // Print system topology
      case 't':
        print_topology = true;
        break;

      // Set verification flag to true
      case 'v':
        verify_ = true;
        break;

      // Collect list of agents involved in bidirectional copy operation
      case 'b':
        status = ParseOptionValue(optarg, bidir_list_);
        if (status) {
          req_copy_bidir_ = REQ_COPY_BIDIR;
          break;
        }
        print_help = true;
        break;

      // Collect list of source pools involved in unidirectional copy operation
      case 's':
        status = ParseOptionValue(optarg, src_list_);
        if (status) {
          req_copy_unidir_ = REQ_COPY_UNIDIR;
          break;
        }
        print_help = true;
        break;

      // Collect list of destination pools involved in unidirectional copy operation
      case 'd':
        status = ParseOptionValue(optarg, dst_list_);
        if (status) {
          req_copy_unidir_ = REQ_COPY_UNIDIR;
          break;
        }
        print_help = true;
        break;

      // Collect request to read a buffer
      case 'r':
        req_read_ = REQ_READ;
        status = ParseOptionValue(optarg, read_list_);
        if (status == false) {
          print_help = true;
        }
        break;

      // Collect request to write a buffer
      case 'w':
        req_write_ = REQ_WRITE;
        status = ParseOptionValue(optarg, write_list_);
        if (status == false) {
          print_help = true;
        }
        break;

      // Size of buffers to use in copy and read/write operations
      case 'm':
        status = ParseOptionValue(optarg, size_list_);
        if (status == false) {
          print_help = true;
        }
        break;

      // Enable Unidirectional copy among all valid pools
      case 'a':
        copy_all_uni = true;
        req_copy_all_unidir_ = REQ_COPY_ALL_UNIDIR;
        break;

      // Enable Bidirectional copy among all valid pools
      case 'A':
        copy_all_bi = true;
        req_copy_all_bidir_ = REQ_COPY_ALL_BIDIR;
        break;

      // getopt implementation returns the value of the unknown
      // option or an option with missing operand in the variable
      // optopt
      case '?':
        std::cout << "Value of optopt is: " << '?' << std::endl;
        if ((optopt == 'b' || optopt == 's' || optopt == 'd' || optopt == 'e')) {
          std::cout << "Error: Option -b -s -d and -e require argument" << std::endl;
        }
        print_help = true;
        break;
      default:
        print_help = true;
        break;
    }
  }
  
  // Print help screen if user option has "-h"
  if (print_help) {
    PrintHelpScreen();
    exit(0);
  }
  
  // Initialize Roc Runtime
  err_ = hsa_init();
  ErrorCheck(err_);

  // Discover the topology of RocR agent in system
  DiscoverTopology();
  
  // Print system topology if user option has "-t"
  if (print_topology) {
    PrintTopology();
    exit(0);
  }

  // Invalidate request if user has requested full
  // copying for both unidirectional and bidirectional
  if ((copy_all_bi) && (copy_all_uni)) {
    PrintHelpScreen();
    exit(0);
  }

  // Initialize pool list if full copying in unidirectional mode is enabled
  if (copy_all_uni) {
    uint32_t size = pool_list_.size();
    for (uint32_t idx = 0; idx < size; idx++) {
      src_list_.push_back(idx);
      dst_list_.push_back(idx);
    }
  }

  // Initialize pool list if full copying in bidirectional mode is enabled
  if (copy_all_bi) {
    uint32_t size = pool_list_.size();
    for (uint32_t idx = 0; idx < size; idx++) {
      bidir_list_.push_back(idx);
    }
  }

  // Initialize the list of buffer sizes to use in copy/read/write operations
  // For All Copy operations use only one buffer size
  if (size_list_.size() == 0) {
    uint32_t size_len = sizeof(SIZE_LIST)/sizeof(uint32_t);
    for (uint32_t idx = 0; idx < size_len; idx++) {
      if ((copy_all_bi) || (copy_all_uni)) {
        if (idx == 0) {
          size_list_.push_back(SIZE_LIST[idx]);
        }
      } else {
        size_list_.push_back(SIZE_LIST[idx]);
      }
    }
  }
  std::sort(size_list_.begin(), size_list_.end());
}

