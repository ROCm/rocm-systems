#ifndef SRC_UTIL_EXCEPTION_H_
#define SRC_UTIL_EXCEPTION_H_

#include <hsa_ven_amd_aqlprofile.h>

#include <exception>
#include <sstream>
#include <string>

#define EXC_ABORT(error, stream)                                                                   \
  {                                                                                                \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    std::cout << oss.str() << std::endl;                                                           \
    abort();                                                                                       \
  }

#define EXC_RAISING(error, stream)                                                                 \
  {                                                                                                \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    throw roctracer::util::exception(error, oss.str());                                            \
  }

#define HIP_EXC_RAISING(error, stream)                                                             \
  {                                                                                                \
    EXC_RAISING(error, "HIP error: " << stream);                                            \
  }

namespace roctracer {
namespace util {

class exception : public std::exception {
 public:
  explicit exception(const uint32_t& status, const std::string& msg) : status_(status), str_(msg) {}
  const char* what() const throw() { return str_.c_str(); }
  uint32_t status() const throw() { return status_; }

 protected:
  const uint32_t status_;
  const std::string str_;
};

}  // namespace util
}  // namespace roctracer

#endif  // SRC_UTIL_EXCEPTION_H_
