/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file slr.hpp
 * @brief Simple Local Runtime (SLR) - Header-only library for single-node
 *        process management and communication for rocSHMEM functional tests.
 *
 * SLR provides a minimal runtime for executing rocSHMEM tests on a single node
 * without requiring MPI or PMIx. It uses fork() for process creation and pipes
 * for inter-process communication.
 *
 * Usage:
 *   ROCSHMEM_SLR_NP=2 ./rocshmem_functional_tests -a 1 -w 1 -z 1
 *
 * Limitations:
 *   - Single node only (no multi-node support)
 *   - Communication only between rank 0 and other ranks
 *   - No direct communication between non-zero ranks
 */

#ifndef _SLR_HPP_
#define _SLR_HPP_

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <vector>
#include <cerrno>
#include <iostream>

namespace SLR {

/**
 * @brief Return codes for SLR operations
 */
enum SLR_Status {
    SLR_SUCCESS = 0,
    SLR_ERROR = -1,
    SLR_NOT_INITIALIZED = -2,
    SLR_ALREADY_INITIALIZED = -3,
    SLR_INVALID_RANK = -4,
    SLR_COMM_ERROR = -5
};

/**
 * @brief Message types for internal communication protocol
 */
enum MessageType {
    MSG_BARRIER = 1,
    MSG_BARRIER_ACK = 2,
    MSG_BROADCAST = 3,
    MSG_INIT = 4,
    MSG_INIT_ACK = 5
};

/**
 * @brief Message header for pipe communication
 */
struct MessageHeader {
    MessageType type;
    size_t payload_size;
};

/**
 * @brief Pipe pair for bidirectional communication
 */
struct PipePair {
    int read_fd;   // File descriptor for reading from child
    int write_fd;  // File descriptor for writing to child
};

/**
 * @brief Global state for the SLR runtime
 */
class SLRContext {
public:
    bool initialized;
    int my_rank;
    int num_procs;
    std::vector<PipePair> pipes;  // Only used by rank 0
    std::vector<pid_t> child_pids;  // Only used by rank 0
    PipePair parent_pipe;  // Only used by non-zero ranks

    SLRContext() : initialized(false), my_rank(-1), num_procs(0) {}
};

// Global context instance
static SLRContext g_slr_ctx;

// Forward declarations
static int SLR_Barrier();

/**
 * @brief Helper function to write data to a pipe with error checking
 */
static int safe_write(int fd, const void* buf, size_t count) {
    size_t total_written = 0;
    const char* ptr = static_cast<const char*>(buf);

    while (total_written < count) {
        ssize_t written = write(fd, ptr + total_written, count - total_written);
        if (written < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted, retry
            }
            std::cerr << "SLR: Write error: " << strerror(errno) << std::endl;
            return SLR_COMM_ERROR;
        }
        total_written += written;
    }
    return SLR_SUCCESS;
}

/**
 * @brief Helper function to read data from a pipe with error checking
 */
static int safe_read(int fd, void* buf, size_t count) {
    size_t total_read = 0;
    char* ptr = static_cast<char*>(buf);

    while (total_read < count) {
        ssize_t nread = read(fd, ptr + total_read, count - total_read);
        if (nread < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted, retry
            }
            std::cerr << "SLR: Read error: " << strerror(errno) << std::endl;
            return SLR_COMM_ERROR;
        }
        if (nread == 0) {
            std::cerr << "SLR: Unexpected EOF on pipe" << std::endl;
            return SLR_COMM_ERROR;
        }
        total_read += nread;
    }
    return SLR_SUCCESS;
}

/**
 * @brief Send a message with optional payload
 */
static int send_message(int fd, MessageType type, const void* payload = nullptr,
                       size_t payload_size = 0) {
    MessageHeader header;
    header.type = type;
    header.payload_size = payload_size;

    int ret = safe_write(fd, &header, sizeof(header));
    if (ret != SLR_SUCCESS) {
        return ret;
    }

    if (payload_size > 0 && payload != nullptr) {
        ret = safe_write(fd, payload, payload_size);
        if (ret != SLR_SUCCESS) {
            return ret;
        }
    }

    return SLR_SUCCESS;
}

/**
 * @brief Receive a message with optional payload
 */
static int recv_message(int fd, MessageType& type, void* payload = nullptr,
                       size_t* payload_size = nullptr) {
    MessageHeader header;

    int ret = safe_read(fd, &header, sizeof(header));
    if (ret != SLR_SUCCESS) {
        return ret;
    }

    type = header.type;

    if (header.payload_size > 0) {
        if (payload_size) {
            *payload_size = header.payload_size;
        }
        if (payload != nullptr) {
            ret = safe_read(fd, payload, header.payload_size);
            if (ret != SLR_SUCCESS) {
                return ret;
            }
        } else {
            // Consume payload bytes to keep pipe aligned, even if caller doesn't want them
            std::vector<char> discard_buffer(header.payload_size);
            ret = safe_read(fd, discard_buffer.data(), header.payload_size);
            if (ret != SLR_SUCCESS) {
                return ret;
            }
        }
    }

    return SLR_SUCCESS;
}

/**
 * @brief Initialize the SLR runtime
 *
 * Rank 0 forks all other ranks, establishes communication channels,
 * assigns ranks, and performs an initial barrier synchronization.
 *
 * @return SLR_SUCCESS on success, error code otherwise
 */
static int SLR_Init() {
    if (g_slr_ctx.initialized) {
        std::cerr << "SLR: Already initialized" << std::endl;
        return SLR_ALREADY_INITIALIZED;
    }

    // Get number of processes from environment variable
    const char* np_env = std::getenv("ROCSHMEM_SLR_NP");
    if (np_env == nullptr) {
        std::cerr << "SLR: ROCSHMEM_SLR_NP environment variable not set" << std::endl;
        return SLR_ERROR;
    }

    int num_procs = std::atoi(np_env);
    if (num_procs <= 0) {
        std::cerr << "SLR: Invalid ROCSHMEM_SLR_NP value: " << np_env << std::endl;
        return SLR_ERROR;
    }

    g_slr_ctx.num_procs = num_procs;

    // Check if we are the initial process or a forked child
    const char* rank_env = std::getenv("_SLR_RANK");

    if (rank_env == nullptr) {
        // We are rank 0 (the initial process)
        g_slr_ctx.my_rank = 0;

        if (num_procs == 1) {
            // Single process case, no forking needed
            g_slr_ctx.initialized = true;
            return SLR_SUCCESS;
        }

        // Reserve space for pipes and child PIDs
        g_slr_ctx.pipes.resize(num_procs);
        g_slr_ctx.child_pids.resize(num_procs, 0);

        // Fork child processes for ranks 1 to num_procs-1
        for (int rank = 1; rank < num_procs; rank++) {
            int pipe_to_child[2];   // Parent writes, child reads
            int pipe_from_child[2]; // Child writes, parent reads

            if (pipe(pipe_to_child) < 0) {
                std::cerr << "SLR: Failed to create pipe to child " << rank << std::endl;
                return SLR_ERROR;
            }

            if (pipe(pipe_from_child) < 0) {
                std::cerr << "SLR: Failed to create pipe from child " << rank << std::endl;
                close(pipe_to_child[0]);
                close(pipe_to_child[1]);
                return SLR_ERROR;
            }

            pid_t pid = fork();

            if (pid < 0) {
                std::cerr << "SLR: Failed to fork child " << rank << std::endl;
                return SLR_ERROR;
            }

            if (pid == 0) {
                // Child process
                // Close unused ends of pipes
                close(pipe_to_child[1]);    // Close write end of pipe to child
                close(pipe_from_child[0]);  // Close read end of pipe from child

                // Close all pipe FDs inherited from previously forked siblings
                // to prevent O(N^2) FD accumulation
                for (int prev_rank = 1; prev_rank < rank; prev_rank++) {
                    close(g_slr_ctx.pipes[prev_rank].read_fd);
                    close(g_slr_ctx.pipes[prev_rank].write_fd);
                }

                // Store parent pipe information
                g_slr_ctx.parent_pipe.read_fd = pipe_to_child[0];
                g_slr_ctx.parent_pipe.write_fd = pipe_from_child[1];

                // Set environment variable to indicate rank for potential re-exec
                char rank_str[32];
                snprintf(rank_str, sizeof(rank_str), "%d", rank);
                setenv("_SLR_RANK", rank_str, 1);

                g_slr_ctx.my_rank = rank;

                // Receive initialization message from parent
                struct {
                    int rank;
                    int num_procs;
                } init_data;

                MessageType msg_type;
                size_t payload_size;
                int ret = recv_message(g_slr_ctx.parent_pipe.read_fd, msg_type,
                                      &init_data, &payload_size);

                if (ret != SLR_SUCCESS || msg_type != MSG_INIT) {
                    std::cerr << "SLR: Child " << rank << " failed to receive init message"
                             << std::endl;
                    _exit(1);
                }

                // Verify rank assignment
                if (init_data.rank != rank || init_data.num_procs != num_procs) {
                    std::cerr << "SLR: Child received incorrect init data" << std::endl;
                    _exit(1);
                }

                // Send acknowledgment
                ret = send_message(g_slr_ctx.parent_pipe.write_fd, MSG_INIT_ACK);
                if (ret != SLR_SUCCESS) {
                    std::cerr << "SLR: Child " << rank << " failed to send init ack"
                             << std::endl;
                    _exit(1);
                }

                g_slr_ctx.initialized = true;

                // Participate in initial barrier
                SLR_Barrier();

                return SLR_SUCCESS;
            }

            // Parent process
            // Close unused ends of pipes
            close(pipe_to_child[0]);    // Close read end of pipe to child
            close(pipe_from_child[1]);  // Close write end of pipe from child

            // Store pipe information for this child
            g_slr_ctx.pipes[rank].write_fd = pipe_to_child[1];
            g_slr_ctx.pipes[rank].read_fd = pipe_from_child[0];
            g_slr_ctx.child_pids[rank] = pid;

            // Send initialization message to child
            struct {
                int rank;
                int num_procs;
            } init_data = {rank, num_procs};

            int ret = send_message(g_slr_ctx.pipes[rank].write_fd, MSG_INIT,
                                  &init_data, sizeof(init_data));
            if (ret != SLR_SUCCESS) {
                std::cerr << "SLR: Failed to send init message to child " << rank
                         << std::endl;
                return SLR_ERROR;
            }

            // Wait for acknowledgment
            MessageType msg_type;
            ret = recv_message(g_slr_ctx.pipes[rank].read_fd, msg_type);
            if (ret != SLR_SUCCESS || msg_type != MSG_INIT_ACK) {
                std::cerr << "SLR: Failed to receive init ack from child " << rank
                         << std::endl;
                return SLR_ERROR;
            }
        }

        g_slr_ctx.initialized = true;

        // Perform initial barrier
        SLR_Barrier();

    } else {
        // This code path is for cases where the child continues in the same binary
        // The actual rank setup happened above in the fork() child branch
        // This is just a safety check
        std::cerr << "SLR: Unexpected initialization path" << std::endl;
        return SLR_ERROR;
    }

    return SLR_SUCCESS;
}

/**
 * @brief Finalize the SLR runtime
 *
 * Performs a barrier synchronization and cleans up resources.
 * Rank 0 waits for all child processes to terminate.
 *
 * @return SLR_SUCCESS on success, error code otherwise
 */
static int SLR_Finalize() {
    if (!g_slr_ctx.initialized) {
        return SLR_NOT_INITIALIZED;
    }

    // Perform final barrier - this is sufficient for synchronization
    int ret = SLR_Barrier();
    if (ret != SLR_SUCCESS) {
        return ret;
    }

    if (g_slr_ctx.my_rank == 0) {
        // Rank 0: Close all pipes
        for (int rank = 1; rank < g_slr_ctx.num_procs; rank++) {
            close(g_slr_ctx.pipes[rank].read_fd);
            close(g_slr_ctx.pipes[rank].write_fd);
        }

        // Wait for all child processes to terminate
        for (int rank = 1; rank < g_slr_ctx.num_procs; rank++) {
            if (g_slr_ctx.child_pids[rank] > 0) {
                int status;
                waitpid(g_slr_ctx.child_pids[rank], &status, 0);
            }
        }

    } else {
        // Non-zero ranks: Close pipes and exit
        close(g_slr_ctx.parent_pipe.read_fd);
        close(g_slr_ctx.parent_pipe.write_fd);

        // Child processes exit here
        _exit(0);
    }

    g_slr_ctx.initialized = false;
    return SLR_SUCCESS;
}

/**
 * @brief Barrier synchronization
 *
 * All processes synchronize at this point. Implemented as:
 * 1. All non-zero ranks send a message to rank 0
 * 2. Rank 0 waits for messages from all ranks
 * 3. Rank 0 sends confirmation to all non-zero ranks
 * 4. All ranks exit after confirmation
 *
 * @return SLR_SUCCESS on success, error code otherwise
 */
static int SLR_Barrier() {
    if (!g_slr_ctx.initialized) {
        return SLR_NOT_INITIALIZED;
    }

    if (g_slr_ctx.num_procs == 1) {
        // Single process, no synchronization needed
        return SLR_SUCCESS;
    }

    if (g_slr_ctx.my_rank == 0) {
        // Rank 0: Wait for barrier messages from all other ranks
        for (int rank = 1; rank < g_slr_ctx.num_procs; rank++) {
            MessageType msg_type;
            int ret = recv_message(g_slr_ctx.pipes[rank].read_fd, msg_type);
            if (ret != SLR_SUCCESS || msg_type != MSG_BARRIER) {
                std::cerr << "SLR: Rank 0 failed to receive barrier from rank " << rank
                         << std::endl;
                return SLR_COMM_ERROR;
            }
        }

        // Send acknowledgment to all ranks
        for (int rank = 1; rank < g_slr_ctx.num_procs; rank++) {
            int ret = send_message(g_slr_ctx.pipes[rank].write_fd, MSG_BARRIER_ACK);
            if (ret != SLR_SUCCESS) {
                std::cerr << "SLR: Rank 0 failed to send barrier ack to rank " << rank
                         << std::endl;
                return SLR_COMM_ERROR;
            }
        }

    } else {
        // Non-zero ranks: Send barrier message to rank 0
        int ret = send_message(g_slr_ctx.parent_pipe.write_fd, MSG_BARRIER);
        if (ret != SLR_SUCCESS) {
            std::cerr << "SLR: Rank " << g_slr_ctx.my_rank
                     << " failed to send barrier message" << std::endl;
            return SLR_COMM_ERROR;
        }

        // Wait for acknowledgment from rank 0
        MessageType msg_type;
        ret = recv_message(g_slr_ctx.parent_pipe.read_fd, msg_type);
        if (ret != SLR_SUCCESS || msg_type != MSG_BARRIER_ACK) {
            std::cerr << "SLR: Rank " << g_slr_ctx.my_rank
                     << " failed to receive barrier ack" << std::endl;
            return SLR_COMM_ERROR;
        }
    }

    return SLR_SUCCESS;
}

/**
 * @brief Get the total number of processes
 *
 * @param[out] size Pointer to store the number of processes
 * @return SLR_SUCCESS on success, error code otherwise
 */
static int SLR_Numprocs(int* size) {
    if (!g_slr_ctx.initialized) {
        return SLR_NOT_INITIALIZED;
    }

    if (size == nullptr) {
        return SLR_ERROR;
    }

    *size = g_slr_ctx.num_procs;
    return SLR_SUCCESS;
}

/**
 * @brief Get the rank of the current process
 *
 * @param[out] rank Pointer to store the rank
 * @return SLR_SUCCESS on success, error code otherwise
 */
static int SLR_Myrank(int* rank) {
    if (!g_slr_ctx.initialized) {
        return SLR_NOT_INITIALIZED;
    }

    if (rank == nullptr) {
        return SLR_ERROR;
    }

    *rank = g_slr_ctx.my_rank;
    return SLR_SUCCESS;
}

/**
 * @brief Broadcast data from rank 0 to all other processes
 *
 * This is a collective operation. All processes must call this function.
 * Only rank 0's buffer content is broadcast to all other processes.
 *
 * @param[in,out] buf Buffer containing data to broadcast (input on rank 0,
 *                    output on other ranks)
 * @param[in] nbytes Number of bytes to broadcast
 * @return SLR_SUCCESS on success, error code otherwise
 */
static int SLR_Broadcast(void* buf, size_t nbytes) {
    if (!g_slr_ctx.initialized) {
        return SLR_NOT_INITIALIZED;
    }

    if (buf == nullptr || nbytes == 0) {
        return SLR_ERROR;
    }

    if (g_slr_ctx.num_procs == 1) {
        // Single process, no broadcast needed
        return SLR_SUCCESS;
    }

    if (g_slr_ctx.my_rank == 0) {
        // Rank 0: Send broadcast data to all other ranks
        for (int rank = 1; rank < g_slr_ctx.num_procs; rank++) {
            int ret = send_message(g_slr_ctx.pipes[rank].write_fd, MSG_BROADCAST,
                                  buf, nbytes);
            if (ret != SLR_SUCCESS) {
                std::cerr << "SLR: Rank 0 failed to send broadcast to rank " << rank
                         << std::endl;
                return SLR_COMM_ERROR;
            }
        }

    } else {
        // Non-zero ranks: Receive broadcast data from rank 0
        MessageType msg_type;
        size_t recv_size;
        int ret = recv_message(g_slr_ctx.parent_pipe.read_fd, msg_type, buf, &recv_size);

        if (ret != SLR_SUCCESS || msg_type != MSG_BROADCAST) {
            std::cerr << "SLR: Rank " << g_slr_ctx.my_rank
                     << " failed to receive broadcast" << std::endl;
            return SLR_COMM_ERROR;
        }

        if (recv_size != nbytes) {
            std::cerr << "SLR: Rank " << g_slr_ctx.my_rank
                     << " received incorrect broadcast size (expected " << nbytes
                     << ", got " << recv_size << ")" << std::endl;
            return SLR_COMM_ERROR;
        }
    }

    return SLR_SUCCESS;
}

} // namespace SLR

#endif /* _SLR_HPP_ */
