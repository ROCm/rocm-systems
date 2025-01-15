// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "fwd.hpp"
#include "module_function.hpp"

#include <timemory/log/color.hpp>
#include <timemory/mpl/policy.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/filepath.hpp>

static inline void
dump_info(std::ostream& _os, const fmodset_t& _data)
{
    module_function::reset_width();
    for(const auto& itr : _data)
        module_function::update_width(itr);

    module_function::write_header(_os);
    for(const auto& itr : _data)
        _os << itr << '\n';

    module_function::reset_width();
}
//
template <typename ArchiveT,
          std::enable_if_t<tim::concepts::is_archive<ArchiveT>::value, int> = 0>
static inline void
dump_info(ArchiveT& _ar, const fmodset_t& _data)
{
    _ar(tim::cereal::make_nvp("module_functions", _data));
}
//
static inline void
dump_info(const string_t& _label, string_t _oname, const string_t& _ext,
          const fmodset_t& _data, int _level, bool _fail)
{
    namespace cereal = tim::cereal;
    namespace policy = tim::policy;

    auto _cfg          = tim::settings::compose_filename_config{};
    _cfg.subdirectory  = "instrumentation";
    _oname             = tim::settings::compose_output_filename(_oname, _ext, _cfg);
    auto _handle_error = [&]() {
        std::stringstream _msg{};
        _msg << "[dump_info] Error opening '" << _oname << " for output";
        verbprintf(_level, "%s\n", _msg.str().c_str());
        if(_fail)
            throw std::runtime_error(std::string{ "[rocprof-sys][exe]" } + _msg.str());
    };

    if(!debug_print && verbose_level < _level) return;

    if(_ext == "txt")
    {
        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, _oname))
            _handle_error();
        else
        {
            verbprintf_bare(_level, "%s", ::tim::log::color::source());
            verbprintf(_level, "Outputting '%s'... ", _oname.c_str());
            dump_info(ofs, _data);
            verbprintf_bare(_level, "Done\n%s", ::tim::log::color::end());
        }
        ofs.close();
    }
    else if(_ext == "xml")
    {
        std::stringstream oss{};
        {
            using output_policy     = policy::output_archive<cereal::XMLOutputArchive>;
            output_policy::indent() = true;
            auto ar                 = output_policy::get(oss);

            ar->setNextName("rocprofsys");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }

        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, _oname))
            _handle_error();
        else
        {
            verbprintf_bare(_level, "%s", ::tim::log::color::source());
            verbprintf(_level, "Outputting '%s'... ", _oname.c_str());
            ofs << oss.str() << std::endl;
            verbprintf_bare(_level, "Done\n%s", ::tim::log::color::end());
        }
        ofs.close();
    }
    else if(_ext == "json")
    {
        std::stringstream oss{};
        {
            using output_policy = policy::output_archive<cereal::PrettyJSONOutputArchive>;
            auto ar             = output_policy::get(oss);

            ar->setNextName("rocprofsys");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }

        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, _oname))
            _handle_error();
        else
        {
            verbprintf_bare(_level, "%s", ::tim::log::color::source());
            verbprintf(_level, "Outputting '%s'... ", _oname.c_str());
            ofs << oss.str() << std::endl;
            verbprintf_bare(_level, "Done\n%s", ::tim::log::color::end());
        }
        ofs.close();
    }
    else
    {
        throw std::runtime_error(TIMEMORY_JOIN(
            "", "[rocprof-sys][exe] Error in ", __FUNCTION__, " :: filename '", _oname,
            "' does not have one of recognized file extensions: txt, json, xml"));
    }
}
//
static inline void
dump_info(const string_t& _oname, const fmodset_t& _data, int _level, bool _fail,
          const string_t& _type, const strset_t& _ext)
{
    for(const auto& itr : _ext)
        dump_info(_type, _oname, itr, _data, _level, _fail);
}
//
static inline void
load_info(const string_t& _label, const string_t& _iname, fmodset_t& _data, int _level)
{
    namespace cereal = tim::cereal;
    namespace policy = tim::policy;

    auto        _pos = _iname.find_last_of('.');
    std::string _ext = {};
    if(_pos != std::string::npos) _ext = _iname.substr(_pos + 1);

    auto _handle_error = [&]() {
        std::stringstream _msg{};
        _msg << "[load_info] Error opening '" << _iname << " for input";
        verbprintf(_level, "%s\n", _msg.str().c_str());
        throw std::runtime_error(std::string{ "[rocprof-sys][exe]" } + _msg.str());
    };

    if(_ext == "xml")
    {
        verbprintf(_level, "Reading '%s'... ", _iname.c_str());
        std::ifstream ifs{ _iname };
        if(!ifs)
            _handle_error();
        else
        {
            using input_policy = policy::input_archive<cereal::XMLInputArchive>;
            auto ar            = input_policy::get(ifs);

            ar->setNextName("rocprofsys");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }
        verbprintf_bare(_level, "Done\n");
        ifs.close();
    }
    else if(_ext == "json")
    {
        verbprintf(_level, "Reading '%s'... ", _iname.c_str());
        std::ifstream ifs{ _iname };
        if(!ifs)
            _handle_error();
        else
        {
            using input_policy = policy::input_archive<cereal::JSONInputArchive>;
            auto ar            = input_policy::get(ifs);

            ar->setNextName("rocprofsys");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }
        verbprintf_bare(_level, "Done\n");
        ifs.close();
    }
    else
    {
        throw std::runtime_error(TIMEMORY_JOIN(
            "", "[rocprof-sys][exe] Error in ", __FUNCTION__, " :: filename '", _iname,
            "' does not have one of recognized extentions: txt, json, xml :: ", _ext));
    }
}
//
static inline void
load_info(const string_t& _inp, std::map<std::string, fmodset_t*>& _data, int _level)
{
    std::vector<std::string> _exceptions{};
    _exceptions.reserve(_data.size());
    for(auto& itr : _data)
    {
        try
        {
            fmodset_t _tmp{};
            load_info(itr.first, _inp, _tmp, _level);
            // add to the existing
            itr.second->insert(_tmp.begin(), _tmp.end());
            // if it did not throw it was successfully loaded
            _exceptions.clear();
            break;
        } catch(std::exception& _e)
        {
            _exceptions.emplace_back(_e.what());
        }
    }
    if(!_exceptions.empty())
    {
        std::stringstream _msg{};
        for(auto& itr : _exceptions)
        {
            _msg << "[rocprof-sys][exe] " << itr << "\n";
        }
        throw std::runtime_error(_msg.str());
    }
}
