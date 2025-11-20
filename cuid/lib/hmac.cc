#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <iostream>
#include "hmac.h"

AMDCUID_HMAC::AMDCUID_HMAC(const std::string& key_file)
    : ctx(nullptr), mac(nullptr), key(nullptr), key_len(0), valid(false)
{
    mac = EVP_MAC_fetch(NULL, "HMAC", NULL);
    if (!mac) {
        std::cerr << "Error creating EVP_MAC" << std::endl;
        return;
    }

    ctx = EVP_MAC_CTX_new(mac);
    if (!ctx) {
        EVP_MAC_free(mac);
        mac = nullptr;
        std::cerr << "Error creating EVP_MAC_CTX" << std::endl;
        return;
    }

    std::ifstream key_file_stream(key_file, std::ios::binary);
    if (!key_file_stream.is_open()) {
        std::cerr << "Error opening key file" << std::endl;
        return;
    }
    key_file_stream.seekg(0, std::ios::end);
    key_len = key_file_stream.tellg();
    key_file_stream.seekg(0, std::ios::beg);
    key = new uint8_t[key_len];
    key_file_stream.read(reinterpret_cast<char*>(key), key_len);
    key_file_stream.close();
    
    valid = true;
}

AMDCUID_HMAC::~AMDCUID_HMAC()
{
    if (ctx) EVP_MAC_CTX_free(ctx);
    if (mac) EVP_MAC_free(mac);
    if (key) delete[] key;
}

amdcuid_status_t AMDCUID_HMAC::generate_hmac_sha256(
    const uint8_t* data,
    size_t data_len,
    uint8_t* out_hash,
    size_t* out_len
)
{
    if (!ctx) {
        std::cerr << "MAC context is not initialized" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    OSSL_PARAM params[2];
    const char* digest_name = "SHA256";
    params[0] = OSSL_PARAM_construct_utf8_string("digest", const_cast<char*>(digest_name), 0);
    params[1] = OSSL_PARAM_construct_end();

    if (!EVP_MAC_CTX_set_params(ctx, params)) {
        std::cerr << "Error setting HMAC digest algorithm" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    if (!EVP_MAC_init(ctx, reinterpret_cast<const unsigned char*>(key), key_len, NULL)) {
        std::cerr << "Error initializing MAC context" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    if (!EVP_MAC_update(ctx, reinterpret_cast<const unsigned char*>(data), data_len)) {
        std::cerr << "Error updating MAC context" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    if (!EVP_MAC_final(ctx, reinterpret_cast<unsigned char*>(out_hash), out_len, EVP_MAX_MD_SIZE)) {
        std::cerr << "Error finalizing MAC" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AMDCUID_HMAC::set_hmac_algorithm(const EVP_MD* md)
{
    if (!ctx) {
        std::cerr << "MAC context is not initialized" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }
    if (!md) {
        md = EVP_sha256();
    }

    OSSL_PARAM params[2];
    const char* algorithm = EVP_MD_get0_name(md);
    params[0] = OSSL_PARAM_construct_utf8_string("digest", const_cast<char*>(algorithm), 0);
    params[1] = OSSL_PARAM_construct_end();

    if (!EVP_MAC_CTX_set_params(ctx, params)) {
        std::cerr << "Error setting HMAC algorithm" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    return AMDCUID_STATUS_SUCCESS;
}
