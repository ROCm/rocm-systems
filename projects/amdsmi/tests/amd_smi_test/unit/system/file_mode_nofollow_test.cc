// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Host-only unit test for amd::smi::SetFileModeNoFollow (no GPU / driver).
// Build/run standalone:
//   c++ -std=c++17 -I projects/amdsmi/rocm_smi/include \
//       projects/amdsmi/tests/amd_smi_test/unit/system/file_mode_nofollow_test.cc \
//       -lgtest -lgtest_main -o t && ./t

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "rocm_smi/rocm_smi_file_utils.h"

using amd::smi::SetFileModeNoFollow;

namespace {

class FileModeNoFollowTest : public ::testing::Test {
 protected:
  char dir_[64] = {};

  void SetUp() override {
    std::strcpy(dir_, "/tmp/rsmi_fmnf_XXXXXX");
    ASSERT_NE(mkdtemp(dir_), nullptr) << std::strerror(errno);
  }

  void TearDown() override {
    for (const char* n : {"f", "decoy", "link", "nope", "fifo"}) {
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
  mode_t      mode;         // mode to apply and expect back
};

TEST_F(FileModeNoFollowTest, SetsModeOnRegularFile) {
  const ModeCase cases[] = {
      {"positive: 0640", 0640},
      {"positive: 0600", 0600},
      {"boundary: 0400 (read-only)", 0400},
      {"corner: 0644", 0644},
  };
  for (const ModeCase& c : cases) {
    const std::string p = path("f");
    const int fd = ::open(p.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0666);
    ASSERT_GE(fd, 0) << c.description << ": " << std::strerror(errno);
    ::close(fd);
    EXPECT_EQ(SetFileModeNoFollow(p.c_str(), c.mode), 0) << c.description;
    EXPECT_EQ(mode_of(p), c.mode) << c.description;
    ::unlink(p.c_str());
  }
}

// security/regression: a symlink swapped in at the path must not redirect the
// mode change to its target. Pre-fix (chmod) follows the symlink; this fails.
TEST_F(FileModeNoFollowTest, RefusesSymlinkAndLeavesTargetUnchanged) {
  const std::string decoy = path("decoy");
  const int fd = ::open(decoy.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  ASSERT_GE(fd, 0) << std::strerror(errno);
  ::close(fd);
  ASSERT_EQ(::chmod(decoy.c_str(), 0644), 0);

  const std::string link = path("link");
  ASSERT_EQ(::symlink(decoy.c_str(), link.c_str()), 0) << std::strerror(errno);

  EXPECT_EQ(SetFileModeNoFollow(link.c_str(), 0600), -1);  // must refuse
  EXPECT_EQ(mode_of(decoy), 0644u);                        // target untouched
}

TEST_F(FileModeNoFollowTest, NonexistentPathFails) {
  EXPECT_EQ(SetFileModeNoFollow(path("nope").c_str(), 0600), -1);
}

// The helper's contract is "existing regular file"; a non-regular path (here a
// FIFO) must be refused rather than fchmod'd, and must not block the open.
TEST_F(FileModeNoFollowTest, RefusesNonRegularFile) {
  const std::string fifo = path("fifo");
  ASSERT_EQ(::mkfifo(fifo.c_str(), 0644), 0) << std::strerror(errno);
  EXPECT_EQ(SetFileModeNoFollow(fifo.c_str(), 0600), -1);
}

}  // namespace
