#ifndef HMAC_H
#define HMAC_H

#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <fstream>
#include <iostream>

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
    int generate_hmac_sha256(const uint8_t* data, size_t data_len, uint8_t* out_hash, size_t* out_len);
    int set_hmac_algorithm(const EVP_MD* md);
};

#endif // HMAC_H