#ifndef SRC_UTIL_EXCEPTION_H_
#define SRC_UTIL_EXCEPTION_H_

#include <exception>
#include <sstream>
#include <string>

#define EXC_ABORT(error, stream)                                                                   \
  do {                                                                                             \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    std::cout << oss.str() << std::endl;                                                           \
    abort();                                                                                       \
  } while (0)

#define EXC_RAISING(error, stream)                                                                 \
  do {                                                                                             \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    throw roctracer::util::exception(error, oss.str());                                            \
  } while (0)

#define HCC_EXC_RAISING(error, stream)                                                             \
  do {                                                                                             \
    EXC_RAISING(error, "HCC error: " << stream);                                                   \
  } while(0)

#define HIP_EXC_RAISING(error, stream)                                                             \
  do {                                                                                             \
    EXC_RAISING(error, "HIP error: " << stream);                                                   \
  } while(0)

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
