NVIDIA cuFile Compatibility
===========================

hipFile / cuFile Differences
----------------------------

In general, hipFile attempts to keep a 1:1 correspondence with cuFile so
that hipifying CUDA code is more straightforward. There are some
differences, however.

* We do not guarantee that specific error codes will be identical between
  cuFile and hipFile (or even that they won't change across hipFile versions,
  for that matter). Robust hipFile error checking should check for any error
  and then optionally drill down into specifics (see the error documentation
  for more details).

* In general, we don't guarantee that any side effects outside of the stated
  effect of an API call will match cuFile's behaviour.

* The specific effects of ``hipFileDriverOpen()``, ``hipFileDriverClose()``, and
  the use count reported by ``hipFileUseCount()``, may not match cuFile's. We
  don't even guarantee that needing to initialize a driver via a special API call
  will be necessary in hipFile.

* hipfile.h does not typedef ``struct sockaddr`` to ``sockaddr_t`` as cufile.h does.

* cufile.h has ``#define cuFileDriverClose cuFileDriverClose_v2``. hipFile has no
  equivalent for this (libcufile does contain both symbols) so both API calls
  will map to ``hipFileDriverClose()``.

* hipfile.h avoids POSIX types that have Windows portability problems:

  - ``long`` is 32 bits on Windows (LLP64) and 64 bits on most POSIX systems
    (LP64). In its place, hipfile.h uses ``int64_t``.
  - hipfile.h uses a platform-independent ``hoff_t`` type in place of ``off_t``
    to avoid problems on Windows, where ``off_t`` is always 32-bit. ``hoff_t``
    is set to ``__int64`` on Windows to match the Windows POSIX layer.
  - ``ssize_t`` is not a standard C type (it's POSIX). There is an easy
    way to map the Windows ``SSIZE_T`` type to ``ssize_t``, though, and we'll
    set that up if we need to support Windows in the future.
