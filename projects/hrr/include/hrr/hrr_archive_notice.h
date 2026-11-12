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
- host buffers copied to or from the device, and host memory registered with
  hipHostRegister: model weights, prompts, inputs and outputs, passwords, keys
  and session tokens, and any personal, customer or third-party data among
  them;
- code objects, fat binary bundles and linker input buffers (the
  application's GPU binary code) and kernel names, which can disclose
  proprietary software;
- kernel argument bytes, other structures passed into HIP calls, and strings
  passed to HIP such as file paths, device names and process ids.
HRR does not intentionally collect user names, host names, command-line
arguments or environment variables as dedicated metadata. However, these
values may still appear in captured memory or other recorded content. HRR does
not encrypt, redact or anonymise anything. File permissions protect the
archive only on the machine that wrote it.

Treat this archive like the most sensitive code and data the process handled.
- Do not post it publicly: not to public repositories, public issues (GitHub
  issues are public), public storage, or CI artefacts readable beyond the
  authorised recipients. Store it only in approved systems with appropriate
  access controls and encryption.
- Share it only with recipients specifically authorised to receive that code
  and data, through an approved secure transfer mechanism, encrypted in
  transit and at rest, and on terms agreed in advance, such as a
  non-disclosure agreement, plus a data processing agreement if it holds
  personal data.
- Before sending, check which calls and kernels it holds with hrr-playback
  --info, and encrypt it for the recipient. If they need a key or passphrase
  to decrypt it, send that separately, through another approved secure
  channel.
- Delete copies you no longer need. If an archive leaks, change every
  password, key and token the process may have held.

You decide what to record and whom to share it with. You are responsible for
having the right to share everything in it, including third-party model
weights and code under their own licences (some require a licence copy and
notice with every transfer), and for data-protection, export and sanctions
rules when it crosses borders.

Replay runs the recorded GPU code on your GPU, with your user's rights: replay
an archive only if you trust whoever produced it.

What HRR is responsible for, and what it leaves to you, is listed under Scope
in projects/hrr/PRIVACY.md of the ROCm source:
https://github.com/ROCm/rocm-systems/blob/develop/projects/hrr/PRIVACY.md

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
