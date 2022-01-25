// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "omnitrace.hpp"

static int  expect_error = NO_ERROR;
static int  error_print  = 0;
static auto regex_opts   = std::regex_constants::egrep | std::regex_constants::optimize;

// set of whole function names to exclude
strset_t
get_whole_function_names()
{
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

//======================================================================================//
//
//  Gets information (line number, filename, and column number) about
//  the instrumented loop and formats it properly.
//
function_signature
get_loop_file_line_info(module_t* mutatee_module, procedure_t* f, flow_graph_t* cfGraph,
                        basic_loop_t* loopToInstrument)
{
    if(!cfGraph || !loopToInstrument || !f) return function_signature{ "", "", "" };

    char        fname[FUNCNAMELEN + 1];
    char        mname[FUNCNAMELEN + 1];
    std::string typeName = {};

    memset(fname, '\0', FUNCNAMELEN + 1);
    memset(mname, '\0', FUNCNAMELEN + 1);

    mutatee_module->getName(mname, FUNCNAMELEN);

    bpvector_t<point_t*>* loopStartInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopStartIter, loopToInstrument);
    bpvector_t<point_t*>* loopExitInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopEndIter, loopToInstrument);

    if(!loopStartInst || !loopExitInst) return function_signature{ "", "", "" };

    unsigned long baseAddr = (unsigned long) (*loopStartInst)[0]->getAddress();
    unsigned long lastAddr =
        (unsigned long) (*loopExitInst)[loopExitInst->size() - 1]->getAddress();
    verbprintf(3, "Loop: size of lastAddr = %lu: baseAddr = %lu, lastAddr = %lu\n",
               (unsigned long) loopExitInst->size(), (unsigned long) baseAddr,
               (unsigned long) lastAddr);

    f->getName(fname, FUNCNAMELEN);

    auto* returnType = f->getReturnType();

    if(returnType)
    {
        typeName = returnType->getName();
    }

    auto*                 params = f->getParams();
    std::vector<string_t> _params;
    if(params)
    {
        for(auto* itr : *params)
        {
            string_t _name = itr->getType()->getName();
            if(_name.empty()) _name = itr->getName();
            _params.push_back(_name);
        }
    }

    bpvector_t<BPatch_statement> lines;
    bpvector_t<BPatch_statement> linesEnd;

    bool info1 = mutatee_module->getSourceLines(baseAddr, lines);

    string_t filename = mname;

    if(info1)
    {
        // filename = lines[0].fileName();
        auto row1 = lines[0].lineNumber();
        auto col1 = lines[0].lineOffset();
        if(col1 < 0) col1 = 0;

        // This following section is attempting to remedy the limitations of
        // getSourceLines for loops. As the program goes through the loop, the resulting
        // lines go from the loop head, through the instructions present in the loop, to
        // the last instruction in the loop, back to the loop head, then to the next
        // instruction outside of the loop. What this section does is starts at the last
        // instruction in the loop, then goes through the addresses until it reaches the
        // next instruction outside of the loop. We then bump back a line. This is not a
        // perfect solution, but we will work with the Dyninst team to find something
        // better.
        bool info2 = mutatee_module->getSourceLines((unsigned long) lastAddr, linesEnd);
        verbprintf(3, "size of linesEnd = %lu\n", (unsigned long) linesEnd.size());

        if(info2)
        {
            auto row2 = linesEnd[0].lineNumber();
            auto col2 = linesEnd[0].lineOffset();
            if(col2 < 0) col2 = 0;
            if(row2 < row1) row1 = row2;  // Fix for wrong line numbers

            return function_signature(typeName, fname, filename, _params, { row1, row2 },
                                      { col1, col2 }, true, info1, info2);
        }
        else
        {
            return function_signature(typeName, fname, filename, _params, { row1, 0 },
                                      { col1, 0 }, true, info1, info2);
        }
    }
    else
    {
        return function_signature(typeName, fname, filename, _params);
    }
}

//======================================================================================//
//
//  We create a new name that embeds the file and line information in the name
//
function_signature
get_func_file_line_info(module_t* mutatee_module, procedure_t* f)
{
    bool          info1, info2;
    unsigned long baseAddr, lastAddr;
    char          fname[FUNCNAMELEN + 1];
    char          mname[FUNCNAMELEN + 1];
    int           row1, col1, row2, col2;
    string_t      filename = {};
    string_t      typeName = {};

    memset(fname, '\0', FUNCNAMELEN + 1);
    memset(mname, '\0', FUNCNAMELEN + 1);

    mutatee_module->getName(mname, FUNCNAMELEN);

    baseAddr = (unsigned long) (f->getBaseAddr());
    f->getAddressRange(baseAddr, lastAddr);
    bpvector_t<BPatch_statement> lines;
    f->getName(fname, FUNCNAMELEN);

    auto* returnType = f->getReturnType();

    if(returnType)
    {
        typeName = returnType->getName();
    }

    auto*                 params = f->getParams();
    std::vector<string_t> _params;
    if(params)
    {
        for(auto* itr : *params)
        {
            string_t _name = itr->getType()->getName();
            if(_name.empty()) _name = itr->getName();
            _params.push_back(_name);
        }
    }

    info1 = mutatee_module->getSourceLines((unsigned long) baseAddr, lines);

    filename = mname;

    if(info1)
    {
        // filename = lines[0].fileName();
        row1 = lines[0].lineNumber();
        col1 = lines[0].lineOffset();

        if(col1 < 0) col1 = 0;
        info2 = mutatee_module->getSourceLines((unsigned long) (lastAddr - 1), lines);
        if(info2)
        {
            row2 = lines[1].lineNumber();
            col2 = lines[1].lineOffset();
            if(col2 < 0) col2 = 0;
            if(row2 < row1) row1 = row2;
            return function_signature(typeName, fname, filename, _params, { row1, 0 },
                                      { 0, 0 }, false, info1, info2);
        }
        else
        {
            return function_signature(typeName, fname, filename, _params, { row1, 0 },
                                      { 0, 0 }, false, info1, info2);
        }
    }
    else
    {
        return function_signature(typeName, fname, filename, _params, { 0, 0 }, { 0, 0 },
                                  false, false, false);
    }
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

    auto _find = [app_image](const string_t& _f) -> procedure_t* {
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

//======================================================================================//
//
bool
c_stdlib_module_constraint(const std::string& _file)
{
    static std::regex _pattern(
        "^(a64l|accept4|alphasort|argp-help|argp-parse|asprintf|atof|atoi|atol|atoll|"
        "auth_des|auth_none|auth_unix|backtrace|backtracesyms|backtracesymsfd|c16rtomb|"
        "cacheinfo|canonicalize|carg|cargf|cargf128|cargl|"
        "catgets|cfmakeraw|cfsetspeed|check_pf|chflags|"
        "clearerr|clearerr_u|clnt_perr|clnt_raw|clnt_tcp|clnt_udp|clnt_unix"
        "settime|copy_file_range|"
        "creat64|ctermid|ctime|ctime_r|ctype|ctype-c99|ctype-c99_l|ctype-extn|ctype_l|"
        "cuserid|daemon|dcigettext|difftime|dirname|div|dl-error|dl-libc|dl-sym|dlerror|"
        "duplocale|dysize|endutxent|envz|epoll_wait|"
        "ether_aton|ether_aton_r|ether_hton|ether_line|ether_ntoa|ether_ntoh|eventfd_"
        "read|eventfd_write|execlp|execv|execvp|explicit_bzero|faccessat|fallocate64|"
        "fattach|fchflags|fchmodat|fdatasync|fdetach|fdopendir|fedisblxcpt|feenablxcpt|"
        "fegetexcept|fegetmode|feholdexcpt|feof_u|ferror_u|fesetenv|fesetexcept|"
        "fesetmode|fesetround|fetestexceptflag|fexecve|ffsll|fgetexcptflg|fgetgrent|"
        "fgetpwent|fgetsgent|fgetspent|fileno|fmemopen|fmtmsg|fnmatch|fprintf|fputc|"
        "fputc_u|fputwc|fputwc_u|freopen|freopen64|fscanf|fseeko|fsetexcptflg|fstab|"
        "fsync|ftello|ftime|ftok|fts|ftw|futimens|futimesat|fwide|fxprintf|gconv_conf|"
        "gconv_db|gconv_dl|genops|getaddrinfo|getaliasent|getaliasent_r|getaliasname|"
        "getauxval|getc|getchar|getchar_u|getdate|getdirentries|getdirname|getentropy|"
        "getenv|getgrent|getgrent_r|getgrgid|getgrnam|gethostid|gethstbyad|gethstbynm|"
        "gethstbynm2|gethstent|gethstent_r|getipv4sourcefilter|getloadavg|getlogin|"
        "getlogin_r|getmsg|getnameinfo|getnetbyad|getnetbynm|getnetent|getnetent_r|"
        "getnetgrent|getnetgrent_r|getopt|getopt1|getpass|getproto|getprtent|getprtent_r|"
        "getprtname|getpwent|getpwent_r|getpwnam|getpwnam_r|getpwuid|getrandom|"
        "getrpcbyname|getrpcbynumber|getrpcent|getrpcent_r|getrpcport|getservent|"
        "getservent_r|getsgent|getsgent_r|getsgnam|getsourcefilter|getspent|getspent_r|"
        "getspnam|getsrvbynm|getsrvbynm_r|getsrvbypt|getsubopt|getsysstats|getttyent|"
        "getusershell|getutent_r|getutline|getutmp|getutxent|getutxid|getutxline|getw|"
        "getwchar|getwchar_u|getwd|glob|gmon|gmtime|grantpt|group_member|gtty|herror|"
        "hsearch|hsearch_r|htons|iconv|iconv_close|iconv_open|idn-stub|if_index|ifaddrs|"
        "inet6_|inet_|inet_|initgroups|insremque|iofgets|iofgetws|iofgetws_u|iofputws|"
        "iofwide|iopopen|ioungetwc|isastream|isctype|isfdtype|key_call|key_prot|killpg|"
        "l64a|labs|lchmod|lckpwdf|lcong48|ldiv|llabs|lldiv|lockf|longjmp|lsearch|lutimes|"
        "makedev|malloc|mblen|mbrtoc16|mbsinit|mbstowcs|mbtowc|mcheck|memccpy|"
        "memchr|memcmp|memfrob|memmem|memset|memstream|mkdtemp|mkfifo|mkfifoat|mkostemp|"
        "mkostemps|mkstemp|mkstemps|mktemp|mlock2|mntent|mntent_r|mpa|"
        "msgctl|msgget|msgsnd|msort|msync|mtrace|netname|nice|nl_langinfo|nsap_addr|nscd_"
        "getgr_r|nscd_gethst_r|nscd_getpw_r|nscd_getserv_r|nscd_helper|"
        "nsswitch|ntp_gettime|ntp_gettimex|obprintf|obstack|oldfmemopen|open_by_handle_"
        "at|opendir|pathconf|pclose|perror|pkey_mprotect|pm_getmaps|pmap_prot|pmap_rmt|"
        "posix_fallocate|posix_fallocate64|preadv64|preadv64v2|printf-prs|printf_fp|"
        "printf_size|profil|psiginfo|psignal|ptrace|ptsname|putc_u|putchar|putchar_u|"
        "putenv|putgrent|putmsg|putpwent|putsgent|putspent|pututxline|putw|putwc_u|"
        "putwchar|putwchar_u|pwritev64|pwritev64v2|raise|rcmd|readv|"
        "reboot|recvfrom|recvmmsg|regex|regexp|remove|rename|renameat|res-close|res_"
        "hconf|res_init|resolv_conf|rexec|rpc_thread|rpmatch|ruserpass|scandir|sched_"
        "cpucount|sched_getaffinity|sched_getcpu|seed48|seekdir|semget|semop|semtimedop|"
        "sendmsg|setbuf|setegid|seteuid|sethostid|setipv4sourcefilter|setlinebuf|"
        "setlogin|setpgrp|setresuid|setrlimit64|setsourcefilter|setutxent|sgetsgent|"
        "sgetspent|shmat|shmdt|shmget|sigandset|sigdelset|siggetmask|sighold|sigignore|"
        "sigintr|sigisempty|signalfd|sigorset|sigpause|sigpending|sigrelse|sigset|"
        "sigstack|sockatmark|speed|splice|sprofil|sscanf|sstk|stime|strcasecmp|"
        "strcasestr|strcat|strchr|strcmp|strcpy|strcspn|strerror|strerror_l|strfmon|"
        "strfromd|strfromf|strfromf128|strfroml|strfry|strlen|strncase|strncat|strncmp|"
        "strncpy|strpbrk|strrchr|strsignal|strspn|strstr|strtod_l|strtof|strtof128_l|"
        "strtof_l|strtoimax|strtok|strtol_l|strtold_l|strtoul|strtoumax|strxfrm|stty|svc|"
        "svc_raw|svc_simple|svc_tcp|svc_udp|svc_unix|swab|sync_file_range|syslog|system|"
        "tcflow|tcflush|tcgetattr|tcgetsid|tcsendbrk|tcsetpgrp|tee|telldir|tempnam|"
        "tmpnam|tmpnam_r|tsearch|ttyname|ttyname_r|ttyslot|tzset|ualarm|ulimit|umount|"
        "unlockpt|updwtmpx|ustat|utimensat|utmp_file|utmpxname|version|"
        "versionsort|vfprintf|vfscanf|vfwscanf|vlimit|vmsplice|vprintf|vtimes|wait[0-9]|"
        "wcfuncs|wcfuncs_l|wcscpy|wcscspn|wcsdup|wcsncat|wcsncmp|wcsnrtombs|wcspbrk|"
        "wcsrchr|wcsstr|wcstod_l|wcstof|wcstoimax|wcstok|wcstold_l|wcstombs|wcstoumax|"
        "wcswidth|wcsxfrm|wctob|wctype_l|wcwidth|wfileops|wgenops|wmemcmp|wmemstream|"
        "wordexp|wstrops|x2y2m1l|xcrypt|xdr|xdr_float|xdr_intXX_t|xdr_mem|xdr_rec|xdr_"
        "ref|xdr_sizeof|xdr_stdio|mq_notify|aio_|timer_routines|nptl-|shm-|sem_close|"
        "setuid|pt-raise|x2y2)",
        regex_opts);

    return std::regex_search(_file, _pattern);
}

//======================================================================================//
//
bool
c_stdlib_function_constraint(const std::string& _func)
{
    static std::regex _pattern(
        "^(malloc|calloc|free|buffer|fscan|fstab|internal|gnu|fprint|isalnum|isalpha|"
        "isascii|isastream|isblank|isblank_l|iscntrl|isctype|isdigit|isdigit_l|isfdtype|"
        "isgraph|islower|islower_l|isprint|isprint_l|ispunct|isspace|isupper|isupper_l|"
        "iswprint|isxdigit|asprintf|atof|atoi|atol|atoll|memalign|memccpy|memcpy|memchr|"
        "memcmp|memfrob|memset|mkdtemp|mkfifo|mkfifoat|mkostemp64|mkostemps64|mkstemp|"
        "mkstemps64|mktemp|mlock2|monstartup|mprobe|mremap_chunk|get_current_dir_name|"
        "get_free_list|getaliasbyname|getaliasent|getauxval|getchar|getchar_unlocked|"
        "getdate|getdirentries|getentropy|getenv|getfs|getgrent|getgrgid|"
        "getgrnam|getgrouplist|gethostbyaddr|gethostbyname|gethostbyname2|gethostent|"
        "gethostid|getifaddrs|getifaddrs_internal|getipv4sourcefilter|getkeyserv_handle|"
        "getloadavg|getlogin|getlogin_fd0|getlogin_r_fd0|getmntent|getmsg|getnetbyaddr|"
        "getnetbyname|getnetent|getnetgrent|getopt|getopt_long|getopt_long_only|getpass|"
        "getprotobyname|getprotobynumber|getprotoent|getpwent|getpwnam|getpwnam_r|"
        "getpwuid|getrandom|getrpcbyname|getrpcbynumber|getrpcent|getrpcport|"
        "getservbyname|getservbyname_r|getservbyport|getservent|getsgent|getsgnam|"
        "getsourcefilter|getspent|getspnam|getsubopt|getttyname|getttyname_r|"
        "getusershell|getutent_r_file|getutent_r_unknown|getutid_r_file|getutid_r_"
        "unknown|getutline|getutline_r_file|getutline_r_unknown|getutmp|getutxent|"
        "getutxid|getutxline|getw|psiginfo|psignal|ptmalloc_init|ptrace|ptsname|putc_"
        "unlocked|putchar|putchar_unlocked|putenv|putgrent|putmsg|putpwent|putsgent|"
        "putspent|pututline_file|pututxline|putw|pw_map_free|pwritev|pwritev2|"
        "qsort|raise|rcmd|re_acquire_state|re_acquire_state_context|re_"
        "comp|re_compile_internal|re_dfa_add_node|re_exec|re_node_set_init_union|re_node_"
        "set_insert|re_node_set_merge|re_search_internal|re_search_stub|re_string_"
        "context_at|re_string_reconstruct|readtcp|readunix|readv|realloc|realpath|str_to_"
        "mpn|strcasecmp|strcat|strcmp|strcpy|strcspn|strerror|strerror_l|strerror_thread_"
        "freeres|strfmon|strfromd|strfromf|strfromf128|strfroml|strfry|strlen|"
        "strncasecmp|strncat|strncmp|strncpy|strpbrk|strrchr|strsignal|strspn|strtof32|"
        "strtoimax|strtok|strtol_l|strtold_l|strtoull|strtoumax|strxfrm|xdrstdio|xdrmem|"
        "inet_|inet6_|clock_|backtrace_|dummy_|fts_|fts64_|fexecv|execv|stime|ftime|"
        "gmtime|wcs|envz_|fmem|fputc|fgetc|fputwc|fgetwc|vprintf|feget|fetest|feenable|"
        "feset|fedisable|nscd_|fork|execl|tzset|ntp_|mtrace|tr_[a-z]+hook|mcheck_[a-z_]+"
        "ftell|fputs|fgets|siglongjmp|sigdelset|killpg|tolower|toupper|daemon|"
        "iconv_[a-z_]+|catopen|catgets|catclose|check_add_mapping$|sem_open|sem_close|"
        "sem_unlink|do_futex_wait|sem_timedwait|unwind_stop|unwind_cleanup|longjmp_"
        "compat|vfork_|elision_init|cr_|cri_|aio_|mq_|sem_init|waitpid$|sigcancel_"
        "handler|sighandler_setxid|start_thread$|clock$|semctl$|shm_open$|shm_unlink$|"
        "printf|dprintf|walker$|clear_once_control$|libcr_|sem_wait$|sem_trywait$|vfork|"
        "pause$|wait$|waitid$|msgrcv$|sigwait$|sigsuspend$|recvmsg$|sendmsg$|"
        "ftrylockfile$|funlockfile$|tee$|setbuf$|setbuffer$|enlarge_userbuf$|convert_and_"
        "print$|feraise|lio_|atomic_|err$|errx$|print_errno_message$|error_tail$|"
        "clntunix_|sem_destroy|setxid_mark_thread|feupdate|send$|connect$|longjmp|pwrite|"
        "accept$|stpncpy$|writeunix$|xflowf$|mbrlen$)",
        regex_opts);

    return std::regex_search(_func, _pattern);
}
//======================================================================================//
//
