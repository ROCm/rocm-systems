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

#include "generate_config.hpp"
#include "common.hpp"
#include "defines.hpp"
#include "info_type.hpp"

#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/policy.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/tpls/cereal/archives.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/tpls/cereal/cereal/archives/json.hpp>
#include <timemory/tpls/cereal/cereal/archives/xml.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/types.hpp>

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <string>

namespace cereal   = ::tim::cereal;
namespace filepath = ::tim::filepath;
using settings     = ::tim::settings;
using ::tim::tsettings;
using ::tim::type_list;
using ::tim::policy::output_archive;

namespace
{
struct custom_setting_serializer
{
    static std::array<bool, TOTAL> options;
};

template <typename Tp>
bool
ignore_setting(const Tp& _v)
{
    if(_v->get_hidden()) return true;
    if(exclude_setting(_v->get_env_name())) return true;
    if(_v->get_config_updated() || _v->get_environ_updated()) return false;
    if(!is_selected(_v->get_env_name()) && !is_selected(_v->get_name())) return true;
    if(available_only && !_v->get_enabled()) return true;
    if(!category_view.empty())
    {
        bool _found = false;
        for(auto& itr : _v->get_categories())
        {
            if(category_view.count(itr) > 0 ||
               category_view.count(TIMEMORY_JOIN("::", "settings", itr)) > 0)
            {
                _found = true;
                break;
            }
        }
        if(!_found) return true;
    }
    if(category_view.count("deprecated") == 0 &&
       category_view.count("settings::deprecated") == 0 &&
       _v->get_categories().count("deprecated") > 0)
        return true;
    if(!print_advanced && category_view.count("advanced") == 0 &&
       category_view.count("settings::advanced") == 0 &&
       _v->get_categories().count("advanced") > 0)
        return true;
    return false;
}
}  // namespace

std::array<bool, TOTAL> custom_setting_serializer::options = { false };

namespace tim
{
namespace operation
{
template <typename Tp>
struct setting_serialization<Tp, custom_setting_serializer>
{
    template <typename ArchiveT>
    void operator()(ArchiveT&, const char*, const Tp&) const
    {}

    template <typename ArchiveT>
    void operator()(ArchiveT&, const char*, Tp&&) const
    {}
};
//
template <typename Tp>
struct setting_serialization<tsettings<Tp>, custom_setting_serializer>
{
    using value_type = tsettings<Tp>;

    template <typename ArchiveT>
    void operator()(ArchiveT& _ar, value_type& _val) const
    {
        static_assert(concepts::is_output_archive<ArchiveT>::value,
                      "Requires an output archive");

        if(ignore_setting(&_val)) return;

        auto _save = std::shared_ptr<value_type>{};
        if constexpr(concepts::is_string_type<Tp>::value)
        {
            if(_val.get_name() != "time_format")
                _val.set(settings::format(_val.get(), settings::instance()->get_tag()));
            if(_val.get_name() == "config_file")
            {
                _save = std::make_shared<value_type>(_val);
                _val.set(Tp{});
            }
        }

        if(all_info)
        {
            _ar(cereal::make_nvp(_val.get_env_name().c_str(), _val));
        }
        else
        {
            Tp _v = _val.get();
            _ar.setNextName(_val.get_env_name().c_str());
            _ar.startNode();
            _ar(cereal::make_nvp("name", _val.get_name()));
            _ar(cereal::make_nvp("value", _v));
            if(custom_setting_serializer::options[DESC])
                _ar(cereal::make_nvp("description", _val.get_description()));
            if(custom_setting_serializer::options[CATEGORY])
                _ar(cereal::make_nvp("description", _val.get_categories()));
            if(custom_setting_serializer::options[VAL])
                _ar(cereal::make_nvp("choices", _val.get_choices()));
            _ar.finishNode();
        }

        if(_save) _val.set(_save->get());
    }
};
}  // namespace operation
}  // namespace tim

template <typename... Tp>
void
push(type_list<Tp...>)
{
    ROCPROFSYS_FOLD_EXPRESSION(
        settings::push_serialize_map_callback<Tp, custom_setting_serializer>());
    ROCPROFSYS_FOLD_EXPRESSION(
        settings::push_serialize_data_callback<Tp, custom_setting_serializer>(
            type_list<std::string>{}));
}

template <typename... Tp>
void
pop(type_list<Tp...>)
{
    ROCPROFSYS_FOLD_EXPRESSION(
        settings::pop_serialize_map_callback<Tp, custom_setting_serializer>());
    ROCPROFSYS_FOLD_EXPRESSION(
        settings::pop_serialize_data_callback<Tp, custom_setting_serializer>(
            type_list<std::string>{}));
}

void
update_choices(const std::shared_ptr<settings>&);

void
generate_config(std::string _config_file, const std::set<std::string>& _config_fmts,
                const std::array<bool, TOTAL>& _options)
{
    custom_setting_serializer::options = _options;

    auto _settings = tim::settings::shared_instance();
    tim::settings::push();
    _settings->find("suppress_config")->second->reset();
    _settings->find("suppress_parsing")->second->reset();

    _config_file   = settings::format(_config_file, _settings->get_tag());
    bool _absolute = _config_file.at(0) == '/';
    auto _dirs     = tim::delimit(_config_file, "/\\/");
    _config_file   = _dirs.back();
    _dirs.pop_back();

    std::string _output_dir = ".";
    if(!_dirs.empty() && !(_dirs.size() == 1 && _dirs.at(0) == "."))
    {
        _output_dir = std::string{ (_absolute) ? "/" : "" } + _dirs.front();
        _dirs.erase(_dirs.begin());
        for(const auto& itr : _dirs)
            _output_dir = TIMEMORY_JOIN('/', _output_dir, itr);
    }
    _output_dir += "/";

    auto        _fmts    = std::set<std::string>{};
    std::string _txt_ext = ".cfg";
    for(std::string itr : { ".cfg", ".txt", ".json", ".xml" })
    {
        if(_config_file.length() <= itr.length()) continue;
        auto _pos = _config_file.rfind(itr);
        if(_pos == _config_file.length() - itr.length())
        {
            if(itr == ".cfg" || itr == ".txt") _txt_ext = itr;
            _fmts.emplace(itr.substr(1));
            _config_file = _config_file.substr(0, _pos);
        }
    }

    if(_fmts.empty() && _config_fmts.size() == 1)
        _fmts = _config_fmts;
    else if(!_fmts.empty())
    {
        for(auto& itr : _config_fmts)
            _fmts.emplace(itr);
    }

    update_choices(_settings);

    using json_t = cereal::PrettyJSONOutputArchive;
    using xml_t  = cereal::XMLOutputArchive;

    // stores the original serializer and replaces it with the custom one
    push(type_list<json_t, xml_t>{});

    static std::time_t _time{ std::time(nullptr) };

    auto _serialize = [_settings](auto&& _ar) {
        _ar->setNextName(TIMEMORY_PROJECT_NAME);
        _ar->startNode();
        (*_ar)(cereal::make_nvp("version", std::string{ ROCPROFSYS_VERSION_STRING }));
        (*_ar)(cereal::make_nvp("date", tim::get_local_datetime("%F_%H.%M", &_time)));
        settings::serialize_settings(*_ar, *_settings);
        _ar->finishNode();
    };

    auto _nout = 0;
    auto _open = [&_nout](std::ofstream& _ofs, const std::string& _fname,
                          const std::string& _type) -> std::ofstream& {
        ++_nout;
        if(file_exists(_fname))
        {
            if(force_config)
            {
                if(settings::verbose() >= 1)
                    std::cout << "[rocprof-sys-avail] File '" << _fname
                              << "' exists. Overwrite force...\n";
            }
            else
            {
                std::cout << "[rocprof-sys-avail] File '" << _fname
                          << "' exists. Overwrite? " << std::flush;
                std::string _response = {};
                std::cin >> _response;
                if(!tim::get_bool(_response, false)) std::exit(EXIT_FAILURE);
            }
        }

        if(filepath::open(_ofs, _fname))
        {
            if(settings::verbose() >= 0)
                printf("[rocprof-sys-avail] Outputting %s configuration file '%s'...\n",
                       _type.c_str(), _fname.c_str());
        }
        else
        {
            throw std::runtime_error(
                TIMEMORY_JOIN(" ", "Error opening", _type, "output file:", _fname));
        }
        return _ofs;
    };

    if(_fmts.count("json") > 0)
    {
        std::stringstream _ss{};
        output_archive<cereal::PrettyJSONOutputArchive>::indent_length() = 4;
        _serialize(output_archive<cereal::PrettyJSONOutputArchive>::get(_ss));
        auto _fname = settings::compose_output_filename(_config_file, ".json", false, -1,
                                                        true, _output_dir);
        std::ofstream ofs{};
        _open(ofs, _fname, "JSON") << _ss.str() << "\n";
    }

    if(_fmts.count("xml") > 0)
    {
        std::stringstream _ss{};
        output_archive<cereal::XMLOutputArchive>::indent() = true;
        _serialize(output_archive<cereal::XMLOutputArchive>::get(_ss));
        auto _fname = settings::compose_output_filename(_config_file, ".xml", false, -1,
                                                        true, _output_dir);
        std::ofstream ofs{};
        _open(ofs, _fname, "XML") << _ss.str() << "\n";
    }

    if(_fmts.count("txt") > 0 || _fmts.count("cfg") > 0 || _nout == 0)
    {
        std::stringstream _ss{};
        size_t            _w = min_width;

        std::vector<std::shared_ptr<tim::vsettings>> _data{};
        for(const auto& itr : *_settings)
        {
            if(exclude_setting(itr.second->get_env_name())) continue;
            for(const auto& citr : itr.second->get_categories())
                if(citr == "deprecated") continue;
            if(ignore_setting(itr.second)) continue;
            _data.emplace_back(itr.second);
        }

        if(alphabetical)
            std::sort(_data.begin(), _data.end(), [](auto _lhs, auto _rhs) {
                return _lhs->get_name() < _rhs->get_name();
            });
        else
        {
            _settings->ordering();
            std::sort(_data.begin(), _data.end(), [](auto _lhs, auto _rhs) {
                auto _lomni = _lhs->get_categories().count("rocprofsys") > 0;
                auto _romni = _rhs->get_categories().count("rocprofsys") > 0;
                if(_lomni && !_romni) return true;
                if(_romni && !_lomni) return false;
                for(const auto* itr :
                    { "ROCPROFSYS_CONFIG", "ROCPROFSYS_MODE", "ROCPROFSYS_TRACE",
                      "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_SAMPLING",
                      "ROCPROFSYS_USE_PROCESS_SAMPLING", "ROCPROFSYS_USE_ROCM",
                      "ROCPROFSYS_USE_AMD_SMI", "ROCPROFSYS_USE_KOKKOSP",
                      "ROCPROFSYS_USE_OMPT", "ROCPROFSYS_USE", "ROCPROFSYS_OUTPUT" })
                {
                    if(_lhs->get_env_name().find(itr) == 0 &&
                       _rhs->get_env_name().find(itr) != 0)
                        return true;
                    if(_rhs->get_env_name().find(itr) == 0 &&
                       _lhs->get_env_name().find(itr) != 0)
                        return false;
                }
                for(const auto* itr :
                    { "ROCPROFSYS_SUPPRESS_PARSING", "ROCPROFSYS_SUPPRESS_CONFIG" })
                {
                    if(_lhs->get_env_name().find(itr) == 0 &&
                       _rhs->get_env_name().find(itr) != 0)
                        return false;
                    if(_rhs->get_env_name().find(itr) == 0 &&
                       _lhs->get_env_name().find(itr) != 0)
                        return true;
                }
                return _lhs->get_name() < _rhs->get_name();
            });
        }

        for(const auto& itr : _data)
        {
            _w = std::max(_w, itr->get_env_name().length());
        }

        for(const auto& itr : _data)
        {
            if(exclude_setting(itr->get_env_name())) continue;

            auto _has_info =
                (all_info || _options[DESC] || _options[CATEGORY] || _options[VAL]);

            if(_has_info) _ss << "\n# name:\n#    " << itr->get_name() << "\n#\n";

            if(_options[DESC] || all_info)
            {
                _ss << "# description:\n";
                auto              _desc = tim::delimit(itr->get_description(), " \n");
                std::stringstream _line{};
                _line << "#   ";
                auto _write = [&_line, &_ss, _w](std::string_view _str) {
                    if(_line.str().length() + _str.length() + 1 >= _w)
                    {
                        _ss << _line.str() << "\n";
                        _line = std::stringstream{};
                        _line << "#   ";
                    }
                    _line << " " << _str;
                };
                for(auto& iitr : _desc)
                    _write(iitr);
                _ss << _line.str() << "\n#\n";
            }
            if(_options[CATEGORY] || all_info)
            {
                _ss << "# categories:\n";
                for(const auto& iitr : itr->get_categories())
                    _ss << "#    " << iitr << "\n";
                _ss << "#\n";
            }
            if((_options[VAL] || all_info) && !itr->get_choices().empty())
            {
                _ss << "# choices:\n";
                for(const auto& iitr : itr->get_choices())
                    _ss << "#    " << iitr << "\n";
                _ss << "#\n";
            }
            if(_has_info) _ss << "\n";
            _ss << std::left << std::setw(_w + 10) << itr->get_env_name() << " = ";
            auto _v = itr->as_string();
            if(itr->get_name() == "config_file") _v = {};
            if(!_v.empty() && expand_keys && itr->get_name() != "time_format")
                _v = settings::format(_v, _settings->get_tag());
            _ss << _v << "\n";
        }
        auto _fname = settings::compose_output_filename(_config_file, _txt_ext, false, -1,
                                                        true, _output_dir);
        std::ofstream ofs{};
        _open(ofs, _fname, "text")
            << "# auto-generated by rocprof-sys-avail (version "
            << ROCPROFSYS_VERSION_STRING << ") on "
            << tim::get_local_datetime("%F @ %H:%M", &_time) << "\n\n"
            << _ss.str();
    }

    // restores the original serializer
    pop(type_list<json_t, xml_t>{});

    tim::settings::pop();
}

void
update_choices(const std::shared_ptr<settings>& _settings)
{
    std::vector<info_type> _info = get_component_info<TIMEMORY_NATIVE_COMPONENTS_END>();

    if(_settings->get_verbose() >= 2 || _settings->get_debug())
        printf("[rocprof-sys-avail] # of component found: %zu\n", _info.size());

    _info.erase(std::remove_if(_info.begin(), _info.end(),
                               [](const auto& itr) {
                                   if(!itr.is_available()) return true;
                                   // NOLINTNEXTLINE
                                   for(const auto& nitr :
                                       { "cuda", "cupti", "nvtx", "roofline", "_bundle",
                                         "data_integer", "data_unsigned", "data_floating",
                                         "printer" })
                                   {
                                       if(itr.name().find(nitr) != std::string::npos)
                                           return true;
                                   }
                                   return false;
                               }),
                _info.end());

    std::vector<std::string> _component_choices = {};
    _component_choices.reserve(_info.size());
    for(const auto& itr : _info)
        _component_choices.emplace_back(itr.id_type());
    if(_settings->get_verbose() >= 2 || _settings->get_debug())
        printf("[rocprof-sys-avail] # of component choices: %zu\n",
               _component_choices.size());
    _settings->find("ROCPROFSYS_TIMEMORY_COMPONENTS")
        ->second->set_choices(_component_choices);
}
