#ifndef SRC_UTIL_EXCEPTION_H_
#define SRC_UTIL_EXCEPTION_H_

#include <exception>
#include <string>
#include <sstream>

#include <hsa_ven_amd_aqlprofile.h>

#define EXC_RAISING(error, stream) { \
  std::ostringstream oss; oss << __FUNCTION__ << "(), " << stream; \
  throw rocprofiler::util::exception(error, oss.str()); \
}

#define AQL_EXC_RAISING(error, stream) { \
  const char* error_string = NULL; \
  const rocprofiler::pfn_t* api = util::HsaRsrcFactory::Instance().AqlProfileApi(); \
  api->hsa_ven_amd_aqlprofile_error_string(&error_string); \
  EXC_RAISING(error, stream << ", " << error_string); \
}

namespace rocprofiler {
namespace util {

class exception : public std::exception {
 public:
  explicit exception(const uint32_t &status, const std::string& msg) : status_(status), str_(msg) {}
  const char* what() const throw() { return str_.c_str(); }
  uint32_t status() const throw() { return status_; }

 protected:
  const uint32_t status_;
  const std::string str_;
};

}  // namespace util
}  // namespace rocprofiler

#endif  // SRC_UTIL_EXCEPTION_H_
