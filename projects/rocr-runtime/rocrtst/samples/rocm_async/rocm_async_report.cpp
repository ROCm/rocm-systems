#include "common.hpp"
#include "rocm_async.hpp"

#include <iomanip>
#include <sstream>
#include <algorithm>

static void printRecord(uint32_t size, double avg_time,
                        double bandwidth, double min_time,
                        double peak_bandwidth) {

  std::stringstream size_str;
  size_str << size << " MB";

  uint32_t format = 15;
  std::cout.precision(3);
  std::cout.width(format);
  std::cout << size_str.str();
  std::cout.width(format);
  std::cout << (avg_time * 1e6);
  std::cout.width(format);
  std::cout << bandwidth;
  std::cout.width(format);
  std::cout << (min_time * 1e6);
  std::cout.width(format);
  std::cout << peak_bandwidth;
  std::cout << std::endl;
}

static void printCopyBanner(uint32_t src_pool_id, uint32_t src_agent_type,
                            uint32_t dst_pool_id, uint32_t dst_agent_type) {

  std::stringstream src_type;
  std::stringstream dst_type;
  (src_agent_type == 0) ? src_type <<  "Cpu" : src_type << "Gpu";
  (dst_agent_type == 0) ? dst_type <<  "Cpu" : dst_type << "Gpu";

  std::cout << std::endl;
  std::cout << "================";
  std::cout << "           Benchmark Result";
  std::cout << "         ================";
  std::cout << std::endl;
  std::cout << "================";
  std::cout << " Src Pool Id: " << src_pool_id;
  std::cout << " Src Agent Type: " << src_type.str();
  std::cout << " ================";
  std::cout << std::endl;
  std::cout << "================";
  std::cout << " Dst Pool Id: " << dst_pool_id;
  std::cout << " Dst Agent Type: " << dst_type.str();
  std::cout << " ================";
  std::cout << std::endl;
  std::cout << std::endl;

  uint32_t format = 15;
  std::cout.setf(ios::left);
  std::cout.width(format);
  std::cout << "Data Size";
  std::cout.width(format);
  std::cout << "Avg Time(us)";
  std::cout.width(format);
  std::cout << "Avg BW(GB/s)";
  std::cout.width(format);
  std::cout << "Min Time(us)";
  std::cout.width(format);
  std::cout << "Peak BW(GB/s)";
  std::cout << std::endl;
}

double RocmAsync::GetMinTime(std::vector<double>& vec) {

  std::sort(vec.begin(), vec.end());
  return vec.at(0);
}

double RocmAsync::GetMeanTime(std::vector<double>& vec) {

  std::sort(vec.begin(), vec.end());
  vec.erase(vec.begin());
  vec.erase(vec.begin(), vec.begin() + num_iteration_ * 0.1);
  vec.erase(vec.begin() + num_iteration_, vec.end());

  double mean = 0.0;
  int num = vec.size();
  for (int it = 0; it < num; it++) {
    mean += vec[it];
  }
  mean /= num;
  return mean;
}

void RocmAsync::Display() const {

  // Iterate through list of transactions and display its timing data
  uint32_t trans_size = trans_list_.size();
  if (trans_size == 0) {
    std::cout << std::endl;
    std::cout << "  One or more of the requests wered filtered out " << std::endl;
    std::cout << "      i.e. No Valid Requests were Made or Remain" << std::endl;
    std::cout << std::endl;
    return;
  }

  if ((req_copy_all_bidir_ == REQ_COPY_ALL_BIDIR) ||
      (req_copy_all_unidir_ == REQ_COPY_ALL_UNIDIR)) {
    DisplayCopyTimeMatrix();
    std::cout << std::endl;
    return;
  }

  for (uint32_t idx = 0; idx < trans_size; idx++) {
    async_trans_t trans = trans_list_[idx];
    if ((trans.req_type_ == REQ_COPY_BIDIR) ||
        (trans.req_type_ == REQ_COPY_UNIDIR)) {
      DisplayCopyTime(trans);
    }
    if ((trans.req_type_ == REQ_READ) ||
        (trans.req_type_ == REQ_WRITE)) {
      DisplayIOTime(trans);
    }
  }
  std::cout << std::endl;
}

void RocmAsync::DisplayIOTime(async_trans_t& trans) const {

}

void RocmAsync::DisplayCopyTime(async_trans_t& trans) const {
  
  // Print Benchmark Header
  uint32_t src_idx = trans.copy.src_idx_;
  uint32_t dst_idx = trans.copy.dst_idx_;
  uint32_t src_dev_idx = pool_list_[src_idx].agent_index_;
  hsa_device_type_t src_dev_type = agent_list_[src_dev_idx].device_type_;
  uint32_t dst_dev_idx = pool_list_[dst_idx].agent_index_;
  hsa_device_type_t dst_dev_type = agent_list_[dst_dev_idx].device_type_;
  printCopyBanner(src_idx, src_dev_type, dst_idx, dst_dev_type);
  
  uint32_t size_len = size_list_.size();
  for (uint32_t idx = 0; idx < size_len; idx++) {
    printRecord(size_list_[idx], trans.avg_time_[idx],
                trans.avg_bandwidth_[idx], trans.min_time_[idx],
                trans.peak_bandwidth_[idx]);
  }
}

void RocmAsync::DisplayCopyTimeMatrix() const {
  
  double* avg_matrix = new double[agent_index_ * agent_index_]();
  double* peak_matrix = new double[agent_index_ * agent_index_]();
  uint32_t trans_size = trans_list_.size();
  for (uint32_t idx = 0; idx < trans_size; idx++) {
    async_trans_t trans = trans_list_[idx];
    uint32_t src_idx = trans.copy.src_idx_;
    uint32_t dst_idx = trans.copy.dst_idx_;
    uint32_t src_dev_idx = pool_list_[src_idx].agent_index_;
    uint32_t dst_dev_idx = pool_list_[dst_idx].agent_index_;
    avg_matrix[(src_dev_idx * agent_index_) + dst_dev_idx] = trans.avg_bandwidth_[0];
    peak_matrix[(src_dev_idx * agent_index_) + dst_dev_idx] = trans.peak_bandwidth_[0];
  }

  uint32_t format = 12;
  std::cout.setf(ios::left);

  std::cout << std::endl;
  std::cout.width(format);
  std::cout << "";
  std::cout.width(format);
  if (req_copy_all_unidir_ == REQ_COPY_ALL_UNIDIR) {
    std::cout << "Peak Bandwidth For Unidirectional Copies GB/sec";
  } else {
    std::cout << "Peak Bandwidth For Bidirectional Copies GB/sec";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout.width(format);
  std::cout << "";
  std::cout.width(format);
  std::cout << "";
  for (uint32_t idx0 = 0; idx0 < agent_index_; idx0++) {
    std::cout.width(format);
    std::stringstream agent_id;
    agent_id << "Dev-" << idx0;
    std::cout << agent_id.str();
  }
  std::cout << std::endl;
  std::cout << std::endl;
  for (uint32_t idx0 = 0; idx0 < agent_index_; idx0++) {
    std::cout.width(format);
    std::cout << "";
    std::stringstream agent_id;
    agent_id << "Dev-" << idx0;
    std::cout.width(format);
    std::cout << agent_id.str();
    for (uint32_t idx1 = 0; idx1 < agent_index_; idx1++) {
      std::cout.width(format);
      std::cout << peak_matrix[(idx0 * agent_index_) + idx1];
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout.width(format);
  std::cout << "";
  std::cout.width(format);
  if (req_copy_all_unidir_ == REQ_COPY_ALL_UNIDIR) {
    std::cout << "Average Bandwidth For Unidirectional Copies GB/sec";
  } else {
    std::cout << "Average Bandwidth For Bidirectional Copies GB/sec";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout.width(format);
  std::cout << "";
  std::cout.width(format);
  std::cout << "";
  for (uint32_t idx0 = 0; idx0 < agent_index_; idx0++) {
    std::cout.width(format);
    std::stringstream agent_id;
    agent_id << "Dev-" << idx0;
    std::cout << agent_id.str();
  }
  std::cout << std::endl;
  std::cout << std::endl;
  for (uint32_t idx0 = 0; idx0 < agent_index_; idx0++) {
    std::cout.width(format);
    std::cout << "";
    std::stringstream agent_id;
    agent_id << "Dev-" << idx0;
    std::cout.width(format);
    std::cout << agent_id.str();
    for (uint32_t idx1 = 0; idx1 < agent_index_; idx1++) {
      std::cout.width(format);
      std::cout << avg_matrix[(idx0 * agent_index_) + idx1];
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /*
  std::cout.width(format);
  std::cout << "";
  std::cout << "@note-1: ZERO in Dev-i != Dev-j means DIRECT PATH doesn't exist";
  std::cout << std::endl;
  std::cout.width(format);
  std::cout << "";
  std::cout << "@note-2: ZERO in Dev-i == Dev-j means COPY operation is filtered out";
  std::cout << std::endl;
  std::cout << std::endl;
  */
}

