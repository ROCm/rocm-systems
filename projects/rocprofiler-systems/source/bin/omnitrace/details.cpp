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

#include "function_signature.hpp"
#include "fwd.hpp"
#include "omnitrace.hpp"

#include <string>
#include <vector>

static int expect_error = NO_ERROR;
static int error_print  = 0;

// set of whole function names to exclude
strset_t
get_whole_function_names()
{
#if 1
    return strset_t{
        "sem_init", "sem_destroy", "sem_open", "sem_close", "sem_post", "sem_wait",
        "sem_getvalue", "sem_clockwait", "sem_timedwait", "sem_trywait", "sem_unlink",
        "fork", "do_futex_wait", "dl_iterate_phdr", "dlinfo", "dlopen", "dlmopen",
        "dlvsym", "dlsym", "getenv", "setenv", "unsetenv", "printf", "fprintf", "fflush",
        "malloc", "malloc_stats", "malloc_trim", "mallopt", "calloc", "free", "pvalloc",
        "valloc", "mmap", "munmap", "fopen", "fclose", "fmemopen", "fmemclose",
        "backtrace", "backtrace_symbols", "backtrace_symbols_fd", "sigaddset",
        "sigandset", "sigdelset", "sigemptyset", "sigfillset", "sighold", "sigisemptyset",
        "sigismember", "sigorset", "sigrelse", "sigvec", "strtok", "strstr", "sbrk",
        "strxfrm", "atexit", "ompt_start_tool", "nanosleep", "cfree", "tolower",
        "toupper", "fileno", "fileno_unlocked", "exit", "quick_exit", "abort",
        // below are functions which never terminate
        "rocr::core::Signal::WaitAny", "rocr::core::Runtime::AsyncEventsLoop",
        "rocr::core::BusyWaitSignal::WaitAcquire",
        "rocr::core::BusyWaitSignal::WaitRelaxed", "rocr::HSA::hsa_signal_wait_scacquire",
        "rocr::os::ThreadTrampoline", "rocr::image::ImageRuntime::CreateImageManager",
        "rocr::AMD::GpuAgent::GetInfo", "rocr::HSA::hsa_agent_get_info", "event_base_loop"
    };
#else
    // should hopefully be removed soon
    return strset_t{ "a64l",
                     "advance",
                     "aio_return",
                     "aio_return64",
                     "argp_error",
                     "argp_failure",
                     "argp_help",
                     "argp_parse",
                     "argp_state_help",
                     "argp_usage",
                     "argz_add",
                     "argz_add_sep",
                     "argz_append",
                     "argz_count",
                     "argz_create",
                     "argz_create_sep",
                     "argz_delete",
                     "argz_extract",
                     "argz_insert",
                     "argz_next",
                     "argz_replace",
                     "argz_stringify",
                     "atexit",
                     "atof",
                     "atoi",
                     "atol",
                     "atoll",
                     "atomic_flag_clear_explicit",
                     "atomic_flag_test_and_set_explicit",
                     "authdes_create",
                     "authdes_getucred",
                     "authdes_pk_create",
                     "authnone_create",
                     "authunix_create",
                     "authunix_create_default",
                     "backtrace",
                     "backtrace_symbols",
                     "backtrace_symbols_fd",
                     "bindresvport",
                     "bindtextdomain",
                     "bind_textdomain_codeset",
                     "bsearch",
                     "btowc",
                     "c16rtomb",
                     "callrpc",
                     "canonicalize_file_name",
                     "catclose",
                     "catgets",
                     "catopen",
                     "cfmakeraw",
                     "cfsetspeed",
                     "chflags",
                     "clearerr",
                     "clearerr_unlocked",
                     "clnt_broadcast",
                     "clnt_create",
                     "clnt_pcreateerror",
                     "clnt_perrno",
                     "clnt_perror",
                     "clntraw_create",
                     "clnt_spcreateerror",
                     "clnt_sperrno",
                     "clnt_sperror",
                     "clnttcp_create",
                     "clntudp_bufcreate",
                     "clntudp_create",
                     "clntunix_create",
                     "confstr",
                     "daemon",
                     "des_setparity",
                     "div",
                     "dlopen",
                     "dlsym",
                     "dlerror",
                     "dladdr",
                     "dlinfo",
                     "dlvsym",
                     "dlmopen",
                     "dl_iterate_phdr",
                     "dysize",
                     "endutxent",
                     "envz_add",
                     "envz_entry",
                     "envz_get",
                     "envz_merge",
                     "envz_remove",
                     "envz_strip",
                     "ether_aton",
                     "ether_hostton",
                     "ether_line",
                     "ether_ntoa",
                     "ether_ntohost",
                     "execl",
                     "execle",
                     "execlp",
                     "execv",
                     "execvp",
                     "execvpe",
                     "explicit_bzero",
                     "fattach",
                     "fclose",
                     "fdetach",
                     "fdopen",
                     "feof_unlocked",
                     "ferror_unlocked",
                     "fflush",
                     "fflush_unlocked",
                     "fgetpos",
                     "fgets",
                     "fgets_unlocked",
                     "fgetws",
                     "fgetws_unlocked",
                     "_fini",
                     "fini",
                     "fmemopen",
                     "fopen",
                     "fopen64",
                     "fopencookie",
                     "fork",
                     "fork_alias",
                     "fork_compat",
                     "fputc_unlocked",
                     "fputs",
                     "fputs_unlocked",
                     "fputwc_unlocked",
                     "fputws",
                     "fputws_unlocked",
                     "fread",
                     "fread_unlocked",
                     "fsetpos",
                     "fsetpos64",
                     "ftell",
                     "fwrite",
                     "fwrite_unlocked",
                     "getdelim",
                     "getgrouplist",
                     "gethostbyname2",
                     "getmntent",
                     "getmsg",
                     "getnetname",
                     "getopt_long",
                     "getopt_long_only",
                     "getpmsg",
                     "getpublickey",
                     "gets",
                     "getsecretkey",
                     "glob_pattern_p",
                     "gnu_dev_major",
                     "gnu_dev_makedev",
                     "gnu_dev_minor",
                     "gnu_get_libc_release",
                     "gnu_get_libc_version",
                     "group_member",
                     "gtty",
                     "hcreate",
                     "hdestroy",
                     "herror",
                     "host2netname",
                     "hsearch",
                     "hstrerror",
                     "htons",
                     "iconv",
                     "iconv_close",
                     "iconv_open",
                     "inet6_opt_append",
                     "inet6_opt_find",
                     "inet6_opt_finish",
                     "inet6_opt_get_val",
                     "inet6_opt_init",
                     "inet6_option_alloc",
                     "inet6_option_append",
                     "inet6_option_find",
                     "inet6_option_init",
                     "inet6_option_next",
                     "inet6_option_space",
                     "inet6_opt_next",
                     "inet6_opt_set_val",
                     "inet6_rth_add",
                     "inet6_rth_getaddr",
                     "inet6_rth_init",
                     "inet6_rth_reverse",
                     "inet6_rth_segments",
                     "inet6_rth_space",
                     "inet_addr",
                     "inet_aton",
                     "inet_lnaof",
                     "inet_makeaddr",
                     "inet_netof",
                     "inet_network",
                     "inet_nsap_addr",
                     "inet_nsap_ntoa",
                     "inet_ntoa",
                     "inet_ntop",
                     "inet_pton",
                     "_init",
                     "init",
                     "initgroups",
                     "initstate",
                     "insque",
                     "iruserok",
                     "iruserok_af",
                     "key_decryptsession",
                     "key_decryptsession_pk",
                     "key_encryptsession",
                     "key_encryptsession_pk",
                     "key_gendes",
                     "key_get_conv",
                     "key_secretkey_is_set",
                     "key_setnet",
                     "key_setsecret",
                     "l64a",
                     "lchmod",
                     "lckpwdf",
                     "lfind",
                     "llabs",
                     "lldiv",
                     "localeconv",
                     "lockf",
                     "lsearch",
                     "mbrtoc16",
                     "mbrtoc32",
                     "mcheck",
                     "mcheck_check_all",
                     "mcheck_pedantic",
                     "mkdtemp",
                     "mkdtemp64",
                     "mkostemp",
                     "mkostemp64",
                     "mkostemps",
                     "mkostemps64",
                     "mkstemp",
                     "mkstemp64",
                     "mkstemps",
                     "mkstemps64",
                     "mktemp",
                     "mktemp64",
                     "moncontrol",
                     "monstartup",
                     "mprobe",
                     "mtrace",
                     "muntrace",
                     "nanosleep",
                     "netname2host",
                     "netname2user",
                     "nl_langinfo",
                     "nl_langinfo_l",
                     "ntohs",
                     "parse_printf_format",
                     "passwd2des",
                     "pclose",
                     "perror",
                     "pmap_getmaps",
                     "pmap_getport",
                     "pmap_rmtcall",
                     "pmap_set",
                     "pmap_unset",
                     "popen",
                     "printf_size",
                     "printf_size_info",
                     "psiginfo",
                     "psignal",
                     "putchar",
                     "putchar_unlocked",
                     "putc_unlocked",
                     "putenv",
                     "putgrent",
                     "putmsg",
                     "putpmsg",
                     "putpwent",
                     "puts",
                     "putsgent",
                     "putspent",
                     "pututxline",
                     "putw",
                     "putwc",
                     "putwchar",
                     "putwchar_unlocked",
                     "putwc_unlocked",
                     "rcmd",
                     "rcmd_af",
                     "reallocarray",
                     "realpath",
                     "re_comp",
                     "re_compile_fastmap",
                     "re_compile_pattern",
                     "re_exec",
                     "regcomp",
                     "regerror",
                     "regexec",
                     "register_printf_modifier",
                     "register_printf_type",
                     "registerrpc",
                     "re_match",
                     "re_match_2",
                     "remque",
                     "re_search",
                     "re_search_2",
                     "re_set_registers",
                     "re_set_syntax",
                     "revoke",
                     "rexec",
                     "rexec_af",
                     "rpmatch",
                     "rresvport",
                     "rresvport_af",
                     "ruserok",
                     "ruserok_af",
                     "ruserpass",
                     "secure_getenv",
                     "seed48",
                     "setbuffer",
                     "setstate",
                     "setvbuf",
                     "sgetsgent",
                     "sgetspent",
                     "sigcancel_handler",
                     "sighandler_setxid",
                     "sstk",
                     "step",
                     "stty",
                     "svcerr_auth",
                     "svcerr_decode",
                     "svcerr_noproc",
                     "svcerr_noprog",
                     "svcerr_progvers",
                     "svcerr_systemerr",
                     "svcerr_weakauth",
                     "svc_exit",
                     "svcfd_create",
                     "svc_getreq",
                     "svc_getreq_common",
                     "svc_getreq_poll",
                     "svc_getreqset",
                     "svcraw_create",
                     "svc_register",
                     "svc_run",
                     "svc_sendreply",
                     "svctcp_create",
                     "svcudp_bufcreate",
                     "svcudp_create",
                     "svcudp_enablecache",
                     "svcunix_create",
                     "svcunixfd_create",
                     "svc_unregister",
                     "swab",
                     "tcgetsid",
                     "tdelete",
                     "tdestroy",
                     "tempnam",
                     "textdomain",
                     "tfind",
                     "thrd_create",
                     "thrd_current",
                     "thrd_detach",
                     "thrd_equal",
                     "thrd_exit",
                     "thrd_join",
                     "thrd_sleep",
                     "thrd_yield",
                     "tmpnam",
                     "tolower",
                     "toupper",
                     "towctrans",
                     "towctrans_l",
                     "tr_break",
                     "tsearch",
                     "tss_create",
                     "tss_delete",
                     "tss_get",
                     "tss_set",
                     "ttyslot",
                     "twalk",
                     "twalk_r",
                     "tzset",
                     "ulckpwdf",
                     "ungetc",
                     "ungetwc",
                     "unwind_stop",
                     "updwtmpx",
                     "user2netname",
                     "utmpname",
                     "utmpxname",
                     "vlimit",
                     "vtimes",
                     "wait",
                     "wait3",
                     "waitpid",
                     "wordexp",
                     "xdecrypt",
                     "xdr_accepted_reply",
                     "xdr_array",
                     "xdr_authdes_cred",
                     "xdr_authdes_verf",
                     "xdr_authunix_parms",
                     "xdr_bool",
                     "xdr_bytes",
                     "xdr_callhdr",
                     "xdr_callmsg",
                     "xdr_char",
                     "xdr_cryptkeyarg",
                     "xdr_cryptkeyarg2",
                     "xdr_cryptkeyres",
                     "xdr_des_block",
                     "xdr_double",
                     "xdr_enum",
                     "xdr_float",
                     "xdr_getcredres",
                     "xdr_hyper",
                     "xdr_int",
                     "xdr_int16_t",
                     "xdr_int64_t",
                     "xdr_int8_t",
                     "xdr_keybuf",
                     "xdr_key_netstarg",
                     "xdr_key_netstres",
                     "xdr_keystatus",
                     "xdr_longlong_t",
                     "xdrmem_create",
                     "xdr_netnamestr",
                     "xdr_netobj",
                     "xdr_opaque",
                     "xdr_opaque_auth",
                     "xdr_pmap",
                     "xdr_pmaplist",
                     "xdr_pointer",
                     "xdr_quad_t",
                     "xdrrec_create",
                     "xdrrec_endofrecord",
                     "xdrrec_eof",
                     "xdrrec_skiprecord",
                     "xdr_reference",
                     "xdr_rejected_reply",
                     "xdr_replymsg",
                     "xdr_rmtcall_args",
                     "xdr_rmtcallres",
                     "xdr_short",
                     "xdr_sizeof",
                     "xdrstdio_create",
                     "xdr_string",
                     "xdr_u_char",
                     "xdr_u_hyper",
                     "xdr_u_int",
                     "xdr_uint16_t",
                     "xdr_uint64_t",
                     "xdr_uint8_t",
                     "xdr_u_long",
                     "xdr_u_longlong_t",
                     "xdr_union",
                     "xdr_unixcred",
                     "xdr_u_quad_t",
                     "xdr_u_short",
                     "xdr_vector",
                     "xdr_void",
                     "xdr_wrapstring",
                     "xencrypt",
                     "xprt_register",
                     "xprt_unregister" };
#endif
}

//======================================================================================//
//
//  Helper functions because the syntax for getting a function or module name is unwieldy
//
std::string_view
get_name(procedure_t* _func)
{
    static auto _v = std::unordered_map<procedure_t*, std::string>{};

    auto itr = _v.find(_func);
    if(itr == _v.end())
    {
        _v.emplace(_func, (_func) ? _func->getDemangledName() : std::string{});
    }

    return _v.at(_func);
}

std::string_view
get_name(module_t* _module)
{
    static auto _v = std::unordered_map<module_t*, std::string>{};

    auto itr = _v.find(_module);
    if(itr == _v.end())
    {
        char _name[FUNCNAMELEN + 1];
        memset(_name, '\0', FUNCNAMELEN + 1);

        if(_module)
        {
            _module->getFullName(_name, FUNCNAMELEN);
            _v.emplace(_module, std::string{ _name });
        }
        else
        {
            _v.emplace(nullptr, std::string{});
        }
    }

    return _v.at(_module);
}

//======================================================================================//
//
//  For selective instrumentation (unused)
//
bool
are_file_include_exclude_lists_empty()
{
    return true;
}

namespace
{
std::string
get_return_type(procedure_t* func)
{
    if(func && func->isInstrumentable() && func->getReturnType())
        return func->getReturnType()->getName();
    return std::string{};
}

auto
get_parameter_types(procedure_t* func)
{
    auto _param_names = std::vector<std::string>{};
    if(func && func->isInstrumentable())
    {
        auto* _params = func->getParams();
        if(_params)
        {
            _param_names.reserve(_params->size());
            for(auto* itr : *_params)
            {
                std::string _name = itr->getType()->getName();
                if(_name.empty()) _name = itr->getName();
                _param_names.emplace_back(_name);
            }
        }
    }
    return _param_names;
}
}  // namespace

//======================================================================================//
//
//  We create a new name that embeds the file and line information in the name
//
function_signature
get_func_file_line_info(module_t* module, procedure_t* func)
{
    using address_t = Dyninst::Address;

    auto _file_name   = get_name(module);
    auto _func_name   = get_name(func);
    auto _return_type = get_return_type(func);
    auto _param_types = get_parameter_types(func);
    auto _base_addr   = address_t{};
    auto _last_addr   = address_t{};
    auto _src_lines   = std::vector<statement_t>{};

    if(func->getAddressRange(_base_addr, _last_addr) &&
       module->getSourceLines(_base_addr, _src_lines) && !_src_lines.empty())
    {
        auto _row = _src_lines.front().lineNumber();
        return function_signature(_return_type, _func_name, _file_name, _param_types,
                                  { _row, 0 }, { 0, 0 }, false, true, false);
    }
    else
    {
        return function_signature(_return_type, _func_name, _file_name, _param_types,
                                  { 0, 0 }, { 0, 0 }, false, false, false);
    }
}

//======================================================================================//
//
//  Gets information (line number, filename, and column number) about
//  the instrumented loop and formats it properly.
//
function_signature
get_loop_file_line_info(module_t* module, procedure_t* func, flow_graph_t*,
                        basic_loop_t* loopToInstrument)
{
    auto basic_blocks = std::vector<BPatch_basicBlock*>{};
    loopToInstrument->getLoopBasicBlocksExclusive(basic_blocks);

    if(basic_blocks.empty()) return function_signature{ "", "", "" };

    auto* _block     = basic_blocks.front();
    auto  _base_addr = _block->getStartAddress();
    auto  _last_addr = _block->getEndAddress();
    for(const auto& itr : basic_blocks)
    {
        if(itr == _block) continue;
        if(itr->dominates(_block))
        {
            _base_addr = itr->getStartAddress();
            _last_addr = itr->getEndAddress();
            _block     = itr;
        }
    }

    auto _file_name   = get_name(module);
    auto _func_name   = get_name(func);
    auto _return_type = get_return_type(func);
    auto _param_types = get_parameter_types(func);
    auto _lines_beg   = std::vector<statement_t>{};
    auto _lines_end   = std::vector<statement_t>{};

    if(module->getSourceLines(_base_addr, _lines_beg))
    {
        // filename = lines[0].fileName();
        int _row1 = 0;
        int _col1 = 0;
        for(auto& itr : _lines_beg)
        {
            if(itr.lineNumber() > 0)
            {
                _row1 = itr.lineNumber();
                _col1 = itr.lineOffset();
                break;
            }
        }

        if(_row1 == 0 && _col1 == 0)
            return function_signature(_return_type, _func_name, _file_name, _param_types);

        int _row2 = 0;
        int _col2 = 0;
        for(auto& itr : _lines_beg)
        {
            _row2 = std::max(_row2, itr.lineNumber());
            _col2 = std::max(_col2, itr.lineOffset());
        }

        if(_col1 < 0) _col1 = 0;

        if(module->getSourceLines(_last_addr, _lines_end))
        {
            for(auto& itr : _lines_end)
            {
                _row2 = std::max(_row2, itr.lineNumber());
                _col2 = std::max(_col2, itr.lineOffset());
            }
            if(_col2 < 0) _col2 = 0;
            if(_row2 < _row1) _row1 = _row2;  // Fix for wrong line numbers

            return function_signature(_return_type, _func_name, _file_name, _param_types,
                                      { _row1, _row2 }, { _col1, _col2 }, true, true,
                                      true);
        }
        else
        {
            return function_signature(_return_type, _func_name, _file_name, _param_types,
                                      { _row1, 0 }, { _col1, 0 }, true, true, false);
        }
    }
    else
    {
        return function_signature(_return_type, _func_name, _file_name, _param_types,
                                  { 0, 0 }, { 0, 0 }, true, false, false);
    }
}

//======================================================================================//
//
//  Gets information (line number, filename, and column number) about
//  the instrumented loop and formats it properly.
//
std::map<basic_block_t*, basic_block_signature>
get_basic_block_file_line_info(module_t* module, procedure_t* func)
{
    std::map<basic_block_t*, basic_block_signature> _data{};
    if(!func) return _data;

    auto* _cfg          = func->getCFG();
    auto  _basic_blocks = std::set<BPatch_basicBlock*>{};
    _cfg->getAllBasicBlocks(_basic_blocks);

    if(_basic_blocks.empty()) return _data;

    auto _file_name   = get_name(module);
    auto _func_name   = get_name(func);
    auto _return_type = get_return_type(func);
    auto _param_types = get_parameter_types(func);

    for(auto&& itr : _basic_blocks)
    {
        auto _base_addr = itr->getStartAddress();
        auto _last_addr = itr->getEndAddress();

        verbprintf(4,
                   "[%s][%s] basic_block: size = %lu: base_addr = %lu, last_addr = %lu\n",
                   _file_name.data(), _func_name.data(),
                   (unsigned long) (_last_addr - _base_addr), _base_addr, _last_addr);

        auto _lines_beg = std::vector<statement_t>{};
        auto _lines_end = std::vector<statement_t>{};

        if(module->getSourceLines(_base_addr, _lines_beg) && !_lines_beg.empty())
        {
            int _row1 = _lines_beg.front().lineNumber();
            int _col1 = _lines_beg.front().lineOffset();

            verbprintf(4, "size of _lines_end = %lu\n",
                       (unsigned long) _lines_end.size());

            if(module->getSourceLines(_last_addr, _lines_end) && !_lines_end.empty())
            {
                int _row2 = _lines_end.back().lineNumber();
                int _col2 = _lines_end.back().lineOffset();

                if(_row2 < _row1) std::swap(_row1, _row2);
                if(_row1 == _row2 && _col2 < _col1) std::swap(_col1, _col2);

                _data.emplace(
                    itr, basic_block_signature{
                             _base_addr, _last_addr,
                             function_signature(_return_type, _func_name, _file_name,
                                                _param_types, { _row1, _row2 },
                                                { _col1, _col2 }, true, true, true) });
            }
            else
            {
                _data.emplace(itr,
                              basic_block_signature{
                                  _base_addr, _last_addr,
                                  function_signature(_return_type, _func_name, _file_name,
                                                     _param_types, { _row1, 0 },
                                                     { _col1, 0 }, true, true, false) });
            }
        }
        else
        {
            _data.emplace(itr, basic_block_signature{
                                   _base_addr, _last_addr,
                                   function_signature(_return_type, _func_name,
                                                      _file_name, _param_types) });
        }
    }

    return _data;
}

//======================================================================================//
//
//  We create a new name that embeds the file and line information in the name
//
std::vector<statement_t>
get_source_code(module_t* module, procedure_t* func)
{
    std::vector<statement_t> _lines{};
    if(!module || !func) return _lines;
    auto*                        _cfg = func->getCFG();
    std::set<BPatch_basicBlock*> _basic_blocks{};
    _cfg->getAllBasicBlocks(_basic_blocks);

    for(auto&& itr : _basic_blocks)
    {
        auto _base_addr = itr->getStartAddress();
        auto _last_addr = itr->getEndAddress();
        for(decltype(_base_addr) _addr = _base_addr; _addr <= _last_addr; ++_addr)
        {
            std::vector<statement_t> _src{};
            if(module->getSourceLines(_addr, _src))
            {
                for(auto&& iitr : _src)
                    _lines.emplace_back(iitr);
            }
        }
    }
    return _lines;
}

//======================================================================================//
//
//  Error callback routine.
//
void
errorFunc(error_level_t level, int num, const char** params)
{
    char line[256];

    const char* msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    if(num != expect_error)
    {
        printf("Error #%d (level %d): %s\n", num, level, line);
        // We consider some errors fatal.
        if(num == 101) exit(-1);
    }
}

//======================================================================================//
//
//  For compatibility purposes
//
procedure_t*
find_function(image_t* app_image, const std::string& _name, const strset_t& _extra)
{
    if(_name.empty()) return nullptr;

    auto _find = [app_image](const std::string& _f) -> procedure_t* {
        // Extract the vector of functions
        bpvector_t<procedure_t*> _found;
        auto* ret = app_image->findFunction(_f.c_str(), _found, false, true, true);
        if(ret == nullptr || _found.empty()) return nullptr;
        return _found.at(0);
    };

    procedure_t* _func = _find(_name);
    auto         itr   = _extra.begin();
    while(_func == nullptr && itr != _extra.end())
    {
        _func = _find(*itr);
        ++itr;
    }

    if(!_func)
    {
        verbprintf(1, "function: '%s' ... not found\n", _name.c_str());
    }
    else
    {
        verbprintf(1, "function: '%s' ... found\n", _name.c_str());
    }

    return _func;
}

//======================================================================================//
//
void
error_func_real(error_level_t level, int num, const char* const* params)
{
    if(num == 0)
    {
        // conditional reporting of warnings and informational messages
        if(error_print > 0)
        {
            if(level == BPatchInfo)
            {
                if(error_print > 1) printf("%s\n", params[0]);
            }
            else
                printf("%s", params[0]);
        }
    }
    else
    {
        // reporting of actual errors
        char        line[256];
        const char* msg = bpatch->getEnglishErrorString(num);
        bpatch->formatErrorString(line, sizeof(line), msg, params);
        if(num != expect_error)
        {
            printf("Error #%d (level %d): %s\n", num, level, line);
            // We consider some errors fatal.
            if(num == 101) exit(-1);
        }
    }
}

//======================================================================================//
//
//  We've a null error function when we don't want to display an error
//
void
error_func_fake(error_level_t level, int num, const char* const* params)
{
    consume_parameters(level, num, params);
    // It does nothing.
}
