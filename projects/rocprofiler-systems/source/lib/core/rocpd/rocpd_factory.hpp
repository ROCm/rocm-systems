#pragma once
#pragma once

#include "data_processor.hpp"
#include "data_storage/database.hpp"

#include <memory>
#include <string>

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
class database;
}

struct data_processor;

class rocpd_factory
{
public:
    static std::shared_ptr<data_storage::database> create_database(int pid)
    {
        return std::make_shared<data_storage::database>(pid);
    }

    static std::shared_ptr<data_processor> create_data_processor(
        std::shared_ptr<data_storage::database> db)
    {
        return std::make_shared<data_processor>(db);
    }

    static std::shared_ptr<data_processor> create_data_processor(int pid)
    {
        auto db = create_database(pid);
        return std::make_shared<data_processor>(db);
    }

private:
    rocpd_factory() = delete;
};

}  // namespace rocpd
}  // namespace rocprofsys
