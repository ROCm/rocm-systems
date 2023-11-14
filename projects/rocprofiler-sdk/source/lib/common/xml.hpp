// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
class Xml
{
public:
    using token_t = std::vector<char>;

    struct level_t;
    using node_vect_t = std::vector<std::shared_ptr<level_t>>;
    using node_list_t = std::list<std::shared_ptr<level_t>>;

    using nodes_t = node_vect_t;
    using opts_t  = std::map<std::string, std::string>;
    struct level_t
    {
        std::string                    tag;
        nodes_t                        nodes;
        opts_t                         opts;
        std::shared_ptr<const level_t> copy;
    };
    using nodes_vec_t = std::vector<std::shared_ptr<level_t>>;
    using map_t       = std::map<std::string, nodes_vec_t>;

    enum
    {
        DECL_STATE,
        BODY_STATE
    };

    static std::shared_ptr<Xml> Create(const std::string& file_name, const Xml* obj = nullptr);

    std::string GetName() { return file_name_; }

    void AddExpr(const std::string& full_tag, const std::string& name, const std::string& expr);
    void AddConst(const std::string& full_tag, const std::string& name, const uint64_t& val);

    nodes_t      GetNodes(const std::string& global_tag) { return (*map_)[global_tag]; }
    const map_t& GetAllNodes() { return (*map_); }

    template <typename Tp>
    Tp ForEach(const Tp& v_i) const;

    struct print_func
    {
        bool operator()(const std::string& global_tag, const std::shared_ptr<level_t>& node);
    };

    void Print() const;

    Xml(std::string file_name, const Xml* obj);
    ~Xml();

private:
    bool        Init();
    void        PreProcess();
    void        Process();
    bool        SpaceCheck() const;
    bool        LineEndCheck();
    token_t     NextToken();
    void        BadFormat(token_t token);
    void        AddLevel(const std::string& tag);
    void        UpLevel();
    void        Copy(const std::shared_ptr<level_t>& from, const std::shared_ptr<level_t>& to);
    void        Inherit(const std::string& tag);
    std::string CurrentLevel() const;
    std::string GlobalTag(const std::string& tag) const;
    void        AddOption(const std::string& key, const std::string& value);
    std::string GetOption(const std::string& key, std::shared_ptr<const level_t> level = nullptr);

    const std::string file_name_;
    unsigned          file_line_{0};
    int               fd_;

    static const size_t kBufSize = 256;
    char                buffer_[kBufSize];

    unsigned                              data_size_{0};
    unsigned                              index_{0};
    unsigned                              state_{0};
    bool                                  comment_{false};
    std::vector<std::shared_ptr<level_t>> stack_;
    bool                                  included_{false};
    std::shared_ptr<level_t>              level_;
    std::shared_ptr<map_t>                map_;
};

template <typename Tp>
Tp
Xml::ForEach(const Tp& v_i) const
{
    Tp v = v_i;
    if(map_)
    {
        for(auto& entry : *map_)
        {
            for(const auto& node : entry.second)
            {
                if(Tp{}(entry.first, node) == false) break;
            }
        }
    }
    return v;
}
}  // namespace common
}  // namespace rocprofiler
