// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "hosttrace.hpp"

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
    if(!cfGraph || !loopToInstrument || !f)
        return function_signature("", "", "");

    char        fname[MUTNAMELEN];
    char        mname[MUTNAMELEN];
    const char* typeName = nullptr;

    mutatee_module->getName(mname, MUTNAMELEN);

    bpvector_t<point_t*>* loopStartInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopStartIter, loopToInstrument);
    bpvector_t<point_t*>* loopExitInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopEndIter, loopToInstrument);

    if(!loopStartInst || !loopExitInst)
        return function_signature("", "", "");

    unsigned long baseAddr = (unsigned long) (*loopStartInst)[0]->getAddress();
    unsigned long lastAddr =
        (unsigned long) (*loopExitInst)[loopExitInst->size() - 1]->getAddress();
    verbprintf(3, "Loop: size of lastAddr = %lu: baseAddr = %lu, lastAddr = %lu\n",
               (unsigned long) loopExitInst->size(), (unsigned long) baseAddr,
               (unsigned long) lastAddr);

    f->getName(fname, MUTNAMELEN);

    auto* returnType = f->getReturnType();

    if(returnType)
    {
        typeName = returnType->getName();
    }
    else
        typeName = "void";

    auto                  params = f->getParams();
    std::vector<string_t> _params;
    if(params)
    {
        for(auto itr : *params)
        {
            string_t _name = itr->getType()->getName();
            if(_name.empty())
                _name = itr->getName();
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
        if(col1 < 0)
            col1 = 0;

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
            if(col2 < 0)
                col2 = 0;
            if(row2 < row1)
                row1 = row2; /* Fix for wrong line numbers*/

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
    char          fname[MUTNAMELEN];
    char          mname[MUTNAMELEN];
    int           row1, col1, row2, col2;
    string_t      filename;
    string_t      typeName;

    mutatee_module->getName(mname, MUTNAMELEN);

    baseAddr = (unsigned long) (f->getBaseAddr());
    f->getAddressRange(baseAddr, lastAddr);
    bpvector_t<BPatch_statement> lines;
    f->getName(fname, MUTNAMELEN);

    auto* returnType = f->getReturnType();

    if(returnType)
    {
        typeName = returnType->getName();
    }
    else
        typeName = "void";

    auto                  params = f->getParams();
    std::vector<string_t> _params;
    if(params)
    {
        for(auto itr : *params)
        {
            string_t _name = itr->getType()->getName();
            if(_name.empty())
                _name = itr->getName();
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

        if(col1 < 0)
            col1 = 0;
        info2 = mutatee_module->getSourceLines((unsigned long) (lastAddr - 1), lines);
        if(info2)
        {
            row2 = lines[1].lineNumber();
            col2 = lines[1].lineOffset();
            if(col2 < 0)
                col2 = 0;
            if(row2 < row1)
                row1 = row2;
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
        if(num == 101)
            exit(-1);
    }
}

//======================================================================================//
//
//  For compatibility purposes
//
procedure_t*
find_function(image_t* app_image, const std::string& _name, strset_t _extra)
{
    if(_name.empty())
        return nullptr;

    auto _find = [app_image](const string_t& _f) -> procedure_t* {
        // Extract the vector of functions
        bpvector_t<procedure_t*> _found;
        auto ret = app_image->findFunction(_f.c_str(), _found, false, true, true);
        if(ret == nullptr || _found.empty())
            return nullptr;
        return _found.at(0);
    };

    procedure_t* _func = _find(_name);
    auto         itr   = _extra.begin();
    while(!_func && itr != _extra.end())
    {
        _func = _find(*itr);
        ++itr;
    }

    if(!_func)
        verbprintf(2, "hosttrace: Unable to find function %s\n", _name.c_str());

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
        if(error_print)
        {
            if(level == BPatchInfo)
            {
                if(error_print > 1)
                    printf("%s\n", params[0]);
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
            if(num == 101)
                exit(-1);
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
inline void
consume()
{
    consume_parameters(initialize_expr, bpatch, use_mpi, stl_func_instr, cstd_func_instr,
                       werror, loop_level_instr, error_print, binary_rewrite, debug_print,
                       expect_error, is_static_exe, available_module_functions,
                       instrumented_module_functions);
}
//
namespace
{
static auto _consumed = (consume(), true);
}
