#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <iostream>
#include "hmac.h"

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

int AMDCUID_HMAC::generate_hmac_sha256(
    const uint8_t* data,
    size_t data_len,
    uint8_t* out_hash,
    size_t* out_len
)
{
    if (!ctx) {
        std::cerr << "MAC context is not initialized" << std::endl;
        // return 12 as the error code so that we can translate that directly to AMDCUID_STATUS_HMAC_ERROR later
        return 12;
    }

    // initialize MAC context with key; other params should have already been given earlier
    if (!EVP_MAC_init(ctx, reinterpret_cast<const unsigned char*>(key), key_len, NULL)) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        std::cerr << "Error initializing MAC context" << std::endl;
        // return 12 as the error code so that we can translate that directly to AMDCUID_STATUS_HMAC_ERROR later
        return 12;
    }

    // update MAC with primary CUID as the message
    if (!EVP_MAC_update(ctx, reinterpret_cast<const unsigned char*>(data), data_len)) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        std::cerr << "Error updating MAC context" << std::endl;
        // return 12 as the error code so that we can translate that directly to AMDCUID_STATUS_HMAC_ERROR later
        return 12;
    }

    // compute HMAC-SHA256 hash
    if (!EVP_MAC_final(ctx, reinterpret_cast<unsigned char*>(out_hash), out_len, EVP_MAX_MD_SIZE)) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        std::cerr << "Error finalizing MAC" << std::endl;
        // return 12 as the error code so that we can translate that directly to AMDCUID_STATUS_HMAC_ERROR later
        return 12;
    }

    return 0;
}

int AMDCUID_HMAC::set_hmac_algorithm(const EVP_MD* md)
{
    if (!ctx) {
        std::cerr << "MAC context is not initialized" << std::endl;
        return 12;
    }
    // if not provided, use SHA256
    if (!md) {
        md = EVP_sha256();
    }

    // set hash algorithm as md
    OSSL_PARAM params[2];
    const char* algorithm = EVP_MD_get0_name(md);
    params[0] = OSSL_PARAM_construct_utf8_string("digest", const_cast<char*>(algorithm), 0);
    params[1] = OSSL_PARAM_construct_end();

    if (!EVP_MAC_CTX_set_params(ctx, params)) {
        std::cerr << "Error setting HMAC algorithm" << std::endl;
        return 12;
    }

    return 0;
}
