// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Host-only unit test for CuidCreateExclusiveFile (no GPU / driver).
// Build/run standalone:
//   c++ -std=c++17 -I projects/cuid/lib \
//       projects/cuid/tests/unit/cuid_exclusive_file_test.cc \
//       -lgtest -lgtest_main -o t && ./t

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "src/cuid_file_utils.h"

namespace {

class CuidCreateExclusiveFileTest : public ::testing::Test {
 protected:
  char dir_[64] = {};

  void SetUp() override {
    std::strcpy(dir_, "/tmp/cuid_exf_XXXXXX");
    ASSERT_NE(mkdtemp(dir_), nullptr) << std::strerror(errno);
  }

  void TearDown() override {
    for (const char* n : {"f", "decoy", "link"}) {
      ::unlink(path(n).c_str());
    }
    ::rmdir(dir_);
  }

  std::string path(const char* name) const { return std::string(dir_) + "/" + name; }

  mode_t mode_of(const std::string& p) const {
    struct stat st {};
    if (::lstat(p.c_str(), &st) != 0) return 0;
    return st.st_mode & 07777;
  }
};

struct ModeCase {
  const char* description;  // what this row exercises + expected outcome
  mode_t      mode;
};

TEST_F(CuidCreateExclusiveFileTest, CreatesFreshFileWithMode) {
  const ModeCase cases[] = {
      {"positive: 0600 (privileged)", 0600},
      {"positive: 0644 (unprivileged)", 0644},
      {"boundary: 0400 (read-only)", 0400},
  };
  for (const ModeCase& c : cases) {
    const std::string p = path("f");
    const int fd = CuidCreateExclusiveFile(p.c_str(), c.mode);
    ASSERT_GE(fd, 0) << c.description;
    ::close(fd);
    EXPECT_EQ(mode_of(p), c.mode) << c.description;
    ::unlink(p.c_str());
  }
}

// security/regression: must refuse a symlink planted at the path (O_NOFOLLOW)
// and must not create/modify its target. Pre-fix (no O_NOFOLLOW) follows it.
TEST_F(CuidCreateExclusiveFileTest, RefusesSymlink) {
  const std::string decoy = path("decoy");  // does not exist yet
  const std::string link = path("link");
  ASSERT_EQ(::symlink(decoy.c_str(), link.c_str()), 0) << std::strerror(errno);

  const int fd = CuidCreateExclusiveFile(link.c_str(), 0600);
  EXPECT_LT(fd, 0);  // must refuse
  if (fd >= 0) ::close(fd);

  struct stat st {};
  EXPECT_NE(::lstat(decoy.c_str(), &st), 0);  // target must not have been created
}

// security: must refuse to reuse a pre-existing regular file (O_EXCL).
TEST_F(CuidCreateExclusiveFileTest, RefusesExistingFile) {
  const std::string p = path("f");
  const int pre = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  ASSERT_GE(pre, 0) << std::strerror(errno);
  ::close(pre);

  const int fd = CuidCreateExclusiveFile(p.c_str(), 0600);
  EXPECT_LT(fd, 0);  // must refuse
  if (fd >= 0) ::close(fd);
  EXPECT_EQ(mode_of(p), 0644u);  // pre-existing file untouched
}

}  // namespace
