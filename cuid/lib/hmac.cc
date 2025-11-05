#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <iostream>
#include "cuid.h"
#include "hmac.h"

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

    // initialize MAC context with key; other params should have already been given earlier
    if (!EVP_MAC_init(ctx, reinterpret_cast<const unsigned char*>(key), key_len, NULL)) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        std::cerr << "Error initializing MAC context" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    // update MAC with primary CUID as the message
    if (!EVP_MAC_update(ctx, reinterpret_cast<const unsigned char*>(data), data_len)) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        std::cerr << "Error updating MAC context" << std::endl;
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    // compute HMAC-SHA256 hash
    if (!EVP_MAC_final(ctx, reinterpret_cast<unsigned char*>(out_hash), out_len, EVP_MAX_MD_SIZE)) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
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
        return AMDCUID_STATUS_HMAC_ERROR;
    }

    return AMDCUID_STATUS_SUCCESS;
}
