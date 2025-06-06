#include "aca_decode.h"

int decode_afid(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision)
{
    if (!register_array)
    {
        return -1;
    }

    aca_raw_data_t raw_data;

    if (array_len == 4) // 32 bytes
    {
        raw_data.aca_status = register_array[0];
        raw_data.aca_addr = register_array[1];
        raw_data.aca_ipid = register_array[2];
        raw_data.aca_synd = register_array[3];
    }
    else if (array_len == 16) // 128 bytes
    {
        raw_data.aca_status = register_array[1];
        raw_data.aca_addr = register_array[2];
        raw_data.aca_ipid = register_array[5];
        raw_data.aca_synd = register_array[6];
    }
    
    else
    {
        return -1; // Unsupported size
    }

    raw_data.flags = flag;
    raw_data.hw_revision = hw_revision;

    aca_error_info_t error_info = aca_decode(&raw_data);
    return error_info.afid;
}

aca_error_info_t decode_error_info(const uint64_t *register_array, size_t array_len, uint32_t flag, uint16_t hw_revision)
{
    aca_raw_data_t raw_data = {0};
    aca_error_info_t error_info = {0};

    if (!register_array)
    {
        return error_info;
    }    if (array_len == 4) // 32 bytes
    {
        raw_data.aca_status = register_array[0];
        raw_data.aca_addr = register_array[1];
        raw_data.aca_ipid = register_array[2];
        raw_data.aca_synd = register_array[3];
    }
    else if (array_len == 16) // 128 bytes
    {
        raw_data.aca_status = register_array[1];
        raw_data.aca_addr = register_array[2];
        raw_data.aca_ipid = register_array[5];
        raw_data.aca_synd = register_array[6];
    }
    else
    {
        return error_info; // Return zero-initialized structure for unsupported size
    }

    raw_data.flags = flag;
    raw_data.hw_revision = hw_revision;

    return aca_decode(&raw_data);
}

