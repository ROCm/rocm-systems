#ifndef HMAC_H
#define HMAC_H

#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <fstream>
#include "cuid.h"

class AMDCUID_HMAC
{
private:
    EVP_MAC_CTX* ctx;
    EVP_MAC* mac;
    uint8_t* key;
    size_t key_len;

public:
    AMDCUID_HMAC(const std::string& key_file);
    ~AMDCUID_HMAC();
    amdcuid_status_t generate_hmac_sha256(const uint8_t* data, size_t data_len, uint8_t* out_hash, size_t* out_len);
    amdcuid_status_t set_hmac_algorithm(const EVP_MD* md);
};

AMDCUID_HMAC::AMDCUID_HMAC(const std::string& key_file)
{
    // set MAC as HMAC
    mac = EVP_MAC_fetch(NULL, "HMAC", NULL);
    if (!mac) {
        std::cerr << "Error creating EVP_MAC" << std::endl;
        return;
    }

    // Create MAC context
    ctx = EVP_MAC_CTX_new(mac);
    if (!ctx) {
        EVP_MAC_free(mac);
        std::cerr << "Error creating EVP_MAC_CTX" << std::endl;
        return;
    }
    // Read key from file
    std::ifstream key_file_stream(key_file, std::ios::binary);
    if (!key_file_stream.is_open())
    {
        std::cerr << "Error opening key file" << std::endl;
        return;
    }
    key_file_stream.seekg(0, std::ios::end);
    key_len = key_file_stream.tellg();
    key_file_stream.seekg(0, std::ios::beg);
    key = new uint8_t[key_len];
    key_file_stream.read(reinterpret_cast<char*>(key), key_len);
    key_file_stream.close();
}

AMDCUID_HMAC::~AMDCUID_HMAC()
{
    EVP_MAC_CTX_free(ctx);
    EVP_MAC_free(mac);
    delete[] key;
}

#endif // HMAC_H