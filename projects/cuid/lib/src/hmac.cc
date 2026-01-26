/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include "hmac.h"

cuid_hmac::cuid_hmac()
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

    std::ifstream key_file_stream(key_file_path, std::ios::binary);
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

cuid_hmac::~cuid_hmac()
{
    if (ctx) EVP_MAC_CTX_free(ctx);
    if (mac) EVP_MAC_free(mac);
    if (key) delete[] key;
}

amdcuid_status_t cuid_hmac::generate_hmac_sha256(
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

amdcuid_status_t cuid_hmac::set_hmac_algorithm(const EVP_MD* md)
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

amdcuid_status_t cuid_hmac::set_hmac_key(const uint8_t key_data[key_length]) {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    if (key) {
        delete[] key;
    }
    key = new uint8_t[key_length];
    std::memcpy(key, key_data, key_length);

    // if key_file exists, delete it first
    if (std::remove(key_file_path.c_str()) != 0 && errno != ENOENT) {
        // failed to delete existing file due to different permissions
        return AMDCUID_STATUS_KEY_ERROR;
    }

    std::ofstream key_file(key_file_path, std::ios::out | std::ios::binary);
    if (!key_file) {
        return AMDCUID_STATUS_KEY_ERROR;
    }
    key_file.write(reinterpret_cast<const char*>(key), key_length);
    if (!key_file) {
        key_file.close();
        return AMDCUID_STATUS_KEY_ERROR;
    }
    key_file.close();

    // set permissions to read/write for owner only
    if (chmod(key_file_path.c_str(), S_IRUSR | S_IWUSR) != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::generate_key(uint8_t key[key_length]) {
    if (!key)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    if (RAND_bytes(key, key_length) != 1) {
        return AMDCUID_STATUS_KEY_ERROR;
    }

    return AMDCUID_STATUS_SUCCESS;
}
