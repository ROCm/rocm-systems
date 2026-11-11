/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <string_view>

/*
 * hrr_archive_notice.h - What an archive holds and how it may be shared.
 *
 * Capture writes kHrrArchiveNotice to NOTICE.txt at the archive root and appends
 * kHrrArchiveNoticeShort to its start notice; hrr-playback prints the short form
 * on stderr. This is the notice's only copy: PRIVACY.md links here, and
 * tools/tests/test_privacy_doc.py fails if it quotes the text.
 */

// NOTICE.txt holds the lines between the two marker lines, which docs can include
// with :start-after: and :end-before: on the markers.
inline constexpr char kHrrArchiveNoticeLines[] =
    /* hrr-archive-notice-begin */ R"notice(
HIP Record & Replay capture archive: confidential, read before sharing

This archive is a record of a process's GPU work. It can contain:
- host memory the process copied to or from the GPU or registered with HIP:
  model weights, prompts, inputs and outputs, passwords, keys and session
  tokens, and any personal, customer or third-party data among them;
- compiled GPU code (code objects, fat binaries, linker inputs) and kernel
  names, which can disclose proprietary software;
- kernel arguments, strings passed to HIP such as file paths, device names and
  process ids.
HRR does not record user names, host names, the command line or environment
variables, but any of them can be inside the recorded memory. HRR does not
encrypt, redact or anonymise anything. File permissions protect the archive
only on the machine that wrote it.

Treat this archive like the most sensitive code and data the process handled.
- Do not post it publicly: no public issues (GitHub issues are public), no
  public storage links, no CI artefacts.
- Share it only with recipients authorised to receive that code and data,
  through a private channel and on terms agreed in advance, such as a
  non-disclosure agreement, plus a data processing agreement if it holds
  personal data.
- Before sending, check which calls and kernels it holds with hrr-playback
  --info, and encrypt it for the recipient.
- Delete copies you no longer need. If an archive leaks, change every
  password, key and token the process may have held.

You decide what to record and whom to share it with. You are responsible for
having the right to share everything in it, including third-party model
weights and code under their own licences (some require a licence copy and
notice with every transfer), and for data-protection, export and sanctions
rules when it crosses borders.

Replay runs the recorded GPU code on your GPU, with your user's rights: replay
an archive only if you trust whoever produced it.

HRR is provided "as is", without warranty of any kind, as stated in its
licence. You, not AMD, are responsible for the content of archives you create
and for how they are shared. Report security issues in HRR to psirt@amd.com.
)notice"; /* hrr-archive-notice-end */
// Without the newline that ends the begin-marker line.
inline constexpr std::string_view kHrrArchiveNotice =
    std::string_view(kHrrArchiveNoticeLines).substr(1);

inline constexpr char kHrrArchiveNoticeShort[] =
    "The archive is unencrypted and confidential: read NOTICE.txt at the archive root "
    "before sharing it.";
