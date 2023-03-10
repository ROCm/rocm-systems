// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gnu/lib-names.h>

#if !defined(LD_LINUX_X86_64_SO)
#    define LD_LINUX_X86_64_SO ""
#endif

#if !defined(LD_SO)
#    define LD_SO ""
#endif

#if !defined(LIBANL_SO)
#    define LIBANL_SO ""
#endif

#if !defined(LIBBROKENLOCALE_SO)
#    define LIBBROKENLOCALE_SO ""
#endif

#if !defined(LIBCRYPT_SO)
#    define LIBCRYPT_SO ""
#endif

#if !defined(LIBC_SO)
#    define LIBC_SO ""
#endif

#if !defined(LIBDL_SO)
#    define LIBDL_SO ""
#endif

#if !defined(LIBGCC_S_SO)
#    define LIBGCC_S_SO ""
#endif

#if !defined(LIBMVEC_SO)
#    define LIBMVEC_SO ""
#endif

#if !defined(LIBM_SO)
#    define LIBM_SO ""
#endif

#if !defined(LIBNSL_SO)
#    define LIBNSL_SO ""
#endif

#if !defined(LIBNSS_COMPAT_SO)
#    define LIBNSS_COMPAT_SO ""
#endif

#if !defined(LIBNSS_DB_SO)
#    define LIBNSS_DB_SO ""
#endif

#if !defined(LIBNSS_DNS_SO)
#    define LIBNSS_DNS_SO ""
#endif

#if !defined(LIBNSS_FILES_SO)
#    define LIBNSS_FILES_SO ""
#endif

#if !defined(LIBNSS_HESIOD_SO)
#    define LIBNSS_HESIOD_SO ""
#endif

#if !defined(LIBNSS_LDAP_SO)
#    define LIBNSS_LDAP_SO ""
#endif

#if !defined(LIBNSS_NISPLUS_SO)
#    define LIBNSS_NISPLUS_SO ""
#endif

#if !defined(LIBNSS_NIS_SO)
#    define LIBNSS_NIS_SO ""
#endif

#if !defined(LIBNSS_TEST1_SO)
#    define LIBNSS_TEST1_SO ""
#endif

#if !defined(LIBNSS_TEST2_SO)
#    define LIBNSS_TEST2_SO ""
#endif

#if !defined(LIBPTHREAD_SO)
#    define LIBPTHREAD_SO ""
#endif

#if !defined(LIBRESOLV_SO)
#    define LIBRESOLV_SO ""
#endif

#if !defined(LIBRT_SO)
#    define LIBRT_SO ""
#endif

#if !defined(LIBTHREAD_DB_SO)
#    define LIBTHREAD_DB_SO ""
#endif

#if !defined(LIBUTIL_SO)
#    define LIBUTIL_SO ""
#endif

#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using func_set_t           = std::unordered_set<std::string>;
using module_func_map_t    = std::unordered_map<std::string, func_set_t>;
using library_module_map_t = std::unordered_map<std::string, module_func_map_t>;

template <typename Tp, typename... TailT>
std::set<Tp>
ordered(const std::unordered_set<Tp, TailT...>&);

template <typename KeyT, typename MappedT, typename... TailT>
std::map<KeyT, MappedT>
ordered(const std::unordered_map<KeyT, MappedT, TailT...>&);

std::optional<std::string> find_library(std::string_view);

std::vector<std::string> find_libraries(std::string_view);

const std::vector<std::string>&
get_library_search_paths();

std::set<std::string>&
get_internal_basic_libs();

std::set<std::string>&
get_internal_libs();

const library_module_map_t&
get_internal_libs_data();

void
parse_internal_libs_data();
