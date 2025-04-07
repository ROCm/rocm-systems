// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/demangle.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <utility>

TEST(common, demangling)
{
    // the purpose of this test is to verify the cxx_demangle function.
    // We want to make sure that the function produces the right demangling
    // when it is a correctly mangled string and does not change the string
    // when demangling fails

    using strview_pair_t = std::pair<std::string_view, std::string_view>;
    for(auto [mangled, demangled] :
        {strview_pair_t{"_ZN11rocprofiler8internal18correlation_config22get_unique_internal_idEv",
                        "rocprofiler::internal::correlation_config::get_unique_internal_id()"},
         strview_pair_t{"_ZN11rocprofiler8internal18get_active_configsEv",
                        "rocprofiler::internal::get_active_configs()"},
         strview_pair_t{"_ZN11rocprofiler8internal22get_registered_configsEv",
                        "rocprofiler::internal::get_registered_configs()"},
         strview_pair_t{
             "_ZZN11rocprofiler8internal18correlation_config22get_unique_internal_idEvE2_v",
             "rocprofiler::internal::correlation_config::get_unique_internal_id()::_v"},
         strview_pair_t{"_ZZN11rocprofiler8internal18get_active_configsEvE2_v",
                        "rocprofiler::internal::get_active_configs()::_v"},
         strview_pair_t{"_ZZN11rocprofiler8internal22get_registered_configsEvE2_v",
                        "rocprofiler::internal::get_registered_configs()::_v"}})
    {
        // verify the demangling works
        EXPECT_EQ(rocprofiler::common::cxx_demangle(mangled), demangled)
            << "failed to demangle '" << mangled << "'";

        // verify we get same string in when improperly mangled string
        auto bad_mangled = std::string{"_Z"} + std::string{mangled};
        EXPECT_EQ(rocprofiler::common::cxx_demangle(bad_mangled), bad_mangled)
            << "demangling succeeded for '" << bad_mangled << "'";
    }
}

TEST(common, truncate)
{
    // this test is verify that the truncate_name function correctly finds the function
    // name in a complex template instantiation

    auto mangled = std::string_view{"_ZSt16__do_uninit_copyIPKSt4pairINSt7__cxx1112basic_"
                                    "stringIcSt11char_traitsIcESaIcEEES6_EPS7_ET0_T_SC_SB_"};

    auto untruncated = std::string_view{
        "std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> "
        ">, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >* "
        "std::__do_uninit_copy<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, "
        "std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, "
        "std::allocator<char> > > const*, std::pair<std::__cxx11::basic_string<char, "
        "std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, "
        "std::char_traits<char>, std::allocator<char> > "
        ">*>(std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, "
        "std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, "
        "std::allocator<char> > > const*, std::pair<std::__cxx11::basic_string<char, "
        "std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, "
        "std::char_traits<char>, std::allocator<char> > > const*, "
        "std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> "
        ">, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >*)"};

    EXPECT_EQ(rocprofiler::common::cxx_demangle(mangled), untruncated);

    EXPECT_EQ(rocprofiler::common::truncate_name(untruncated),
              std::string_view{"__do_uninit_copy"});

    EXPECT_EQ(rocprofiler::common::truncate_name(rocprofiler::common::cxx_demangle(mangled)),
              std::string_view{"__do_uninit_copy"});
}
