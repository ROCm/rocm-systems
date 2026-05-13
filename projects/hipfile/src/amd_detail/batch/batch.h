/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}
namespace hipFile {
class ITaskGroup;
}

namespace hipFile {

struct InvalidBatchHandle : public std::invalid_argument {
    InvalidBatchHandle() : std::invalid_argument{"Invalid batch handle"}
    {
    }
};

struct BatchFull : public std::invalid_argument {
    BatchFull() : std::invalid_argument{"Not enough room in batch"}
    {
    }
};

struct InvalidStateTransition : public std::logic_error {
    InvalidStateTransition(const char *from, const char *to);
};

namespace batchOperationState {

    struct Waiting;
    struct Pending;
    struct Running;
    struct Complete;
    struct Canceled;
    struct Invalid;
    struct Timeout;
    struct Failed;

    template <class Derived> struct StateBase {
        ssize_t ret() const noexcept
        {
            return 0;
        }

        bool isTerminal() const noexcept
        {
            return false;
        }

        template <class To> [[noreturn]] To transitionTo(const To &to) const
        {
            throw InvalidStateTransition{self().name(), to.name()};
        }

    private:
        const Derived &self() const noexcept
        {
            return static_cast<const Derived &>(*this);
        }
    };

    struct Waiting : StateBase<Waiting> {
        using StateBase<Waiting>::transitionTo;

        const char *name() const noexcept
        {
            return "hipFileWaiting";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFileWaiting;
        }

        Pending transitionTo(const Pending &next) const;
        Invalid transitionTo(const Invalid &next) const;
        Failed  transitionTo(const Failed &next) const;
    };

    struct Pending : StateBase<Pending> {
        using StateBase<Pending>::transitionTo;

        const char *name() const noexcept
        {
            return "hipFilePending";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFilePending;
        }

        Running  transitionTo(const Running &next) const;
        Canceled transitionTo(const Canceled &next) const;
        Failed   transitionTo(const Failed &next) const;
    };

    struct Running : StateBase<Running> {
        using StateBase<Running>::transitionTo;

        const char *name() const noexcept
        {
            return "hipFileRunning";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFilePending;
        }

        Complete transitionTo(const Complete &next) const;
        Failed   transitionTo(const Failed &next) const;
        Timeout  transitionTo(const Timeout &next) const;
    };

    struct Complete : StateBase<Complete> {
        using StateBase<Complete>::transitionTo;

        explicit Complete(ssize_t _num_bytes) : num_bytes{_num_bytes}
        {
        }

        const char *name() const noexcept
        {
            return "hipFileComplete";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFileComplete;
        }

        ssize_t ret() const noexcept
        {
            return num_bytes;
        }

        bool isTerminal() const noexcept
        {
            return true;
        }

        Failed transitionTo(const Failed &next) const;

        ssize_t num_bytes;
    };

    struct Canceled : StateBase<Canceled> {
        using StateBase<Canceled>::transitionTo;

        const char *name() const noexcept
        {
            return "hipFileCanceled";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFileCanceled;
        }

        bool isTerminal() const noexcept
        {
            return true;
        }

        Canceled transitionTo(const Canceled &) const;
        Failed   transitionTo(const Failed &next) const;
    };

    struct Invalid : StateBase<Invalid> {
        using StateBase<Invalid>::transitionTo;

        const char *name() const noexcept
        {
            return "hipFileInvalid";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFileInvalid;
        }

        bool isTerminal() const noexcept
        {
            return true;
        }

        Failed transitionTo(const Failed &next) const;
    };

    struct Timeout : StateBase<Timeout> {
        using StateBase<Timeout>::transitionTo;

        const char *name() const noexcept
        {
            return "hipFileTimeout";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFileTimeout;
        }

        bool isTerminal() const noexcept
        {
            return true;
        }

        Failed transitionTo(const Failed &next) const;
    };

    struct Failed : StateBase<Failed> {
        using StateBase<Failed>::transitionTo;

        explicit Failed(ssize_t _error) : error{_error}
        {
        }

        const char *name() const noexcept
        {
            return "hipFileFailed";
        }

        hipFileStatus_t toPublic() const noexcept
        {
            return hipFileFailed;
        }

        ssize_t ret() const noexcept
        {
            return error;
        }

        bool isTerminal() const noexcept
        {
            return true;
        }

        Failed transitionTo(const Failed &next) const;

        ssize_t error;
    };

    using OperationState =
        std::variant<Waiting, Pending, Running, Complete, Canceled, Invalid, Timeout, Failed>;
}

/// @brief Represents a single IO Request
class IBatchOperation {
public:
    virtual ~IBatchOperation() = default;

    /// @brief Mark the operation as accepted and ready to run.
    virtual void markPending() = 0;

    /// @brief Cancel the operation if it can be transitioned to Canceled; otherwise no-op.
    virtual void tryCancel() = 0;

    /// @brief Execute the operation.
    virtual void run() noexcept = 0;

    /// @brief Record an internal execution failure on the operation.
    virtual void recordInternalError() = 0;

    /// @brief Return a snapshot of the operation event state.
    virtual hipFileIOEvents_t event() const = 0;

    /// @brief Return whether the operation has reached a terminal status.
    virtual bool isTerminal() const = 0;
};

/// @brief Represents a single IO Request
class BatchOperation : public IBatchOperation {
public:
    // Don't allow copying
    BatchOperation(const BatchOperation &)            = delete;
    BatchOperation &operator=(const BatchOperation &) = delete;

    // Don't allow moving
    BatchOperation(BatchOperation &&)            = delete;
    BatchOperation &operator=(BatchOperation &&) = delete;
    ~BatchOperation() override                   = default;

    /// @brief Create an operation to handle and track an IO request.
    /// @param [in] params IO parameters
    /// @param [in] buffer Buffer corresponding to params->u.batch.devPtr_base
    /// @param [in] file File corresponding params->fh
    BatchOperation(std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                   std::shared_ptr<IFile> file);

    /// @brief Mark the operation as accepted and ready to run.
    void markPending() override;

    /// @brief Cancel the operation if it can be transitioned to Canceled; otherwise no-op.
    void tryCancel() override;

    /// @brief Execute the operation.
    void run() noexcept override;

    /// @brief Record an internal execution failure on the operation.
    void recordInternalError() override;

    /// @brief Return a snapshot of the operation event state.
    hipFileIOEvents_t event() const override;

    /// @brief Return whether the operation has reached a terminal status.
    bool isTerminal() const override;

private:
    /// @brief A copy of the params provided by the application.
    /// @internal Keep this listed at the top of BatchOperation.
    const std::unique_ptr<const hipFileIOParams_t> io_params;

    /// @brief A reference to the specified Buffer.
    const std::shared_ptr<IBuffer> buffer;

    /// @brief A reference to the specified registered File.
    const std::shared_ptr<IFile> file;

    /// @brief Protects operation state.
    mutable std::mutex state_mutex;

    /// @brief Current operation state.
    batchOperationState::OperationState state{batchOperationState::Waiting{}};

    /// @brief Move to the next operation state. Caller must hold state_mutex.
    template <class Next> void transitionTo(Next next);
};

class IBatchOperationFactory {
public:
    virtual ~IBatchOperationFactory() = default;

    virtual std::shared_ptr<IBatchOperation> create(std::unique_ptr<const hipFileIOParams_t> params,
                                                    std::shared_ptr<IBuffer>                 buffer,
                                                    std::shared_ptr<IFile>                   file) = 0;
};

class BatchOperationFactory : public IBatchOperationFactory {
public:
    std::shared_ptr<IBatchOperation> create(std::unique_ptr<const hipFileIOParams_t> params,
                                            std::shared_ptr<IBuffer>                 buffer,
                                            std::shared_ptr<IFile>                   file) override;
};

class IBatchContext {
public:
    static constexpr unsigned MAX_SIZE = 128;

    virtual ~IBatchContext()                                                               = default;
    virtual unsigned getCapacity() const noexcept                                          = 0;
    virtual void     submitOperations(const hipFileIOParams_t *params, unsigned num_params,
                                      IBatchOperationFactory *operation_factory = nullptr) = 0;
    virtual void     getStatus(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp,
                               struct timespec *timeout)                                   = 0;
    virtual void     cancelOperations()                                                    = 0;
};

class BatchContext : public IBatchContext, public std::enable_shared_from_this<BatchContext> {
public:
    // Don't allow copying
    BatchContext(const BatchContext &)            = delete;
    BatchContext &operator=(const BatchContext &) = delete;

    // Don't allow moving
    BatchContext(BatchContext &&)            = delete;
    BatchContext &operator=(BatchContext &&) = delete;

    ~BatchContext() override;

    ///
    /// @brief Return the max number of concurrent operations supported by this BatchContext.
    ///
    /// @return The max number of concurrent operations that can be processed by this BatchContext.
    /// @note This may not exceed the value returned by `MAX_SIZE`.
    unsigned getCapacity() const noexcept override;

    ///
    /// @brief Submit one or more operations to this Context.
    /// @param [in] params     Pointer to the operations to enqueue.
    /// @param [in] num_params Number of operations to enqueue.
    ///
    /// @note This is an All or None operation. If one submitted operation is not valid, no operations
    ///       will be submitted.
    ///
    void submitOperations(const hipFileIOParams_t *params, const unsigned num_params,
                          IBatchOperationFactory *operation_factory = nullptr) override;

    ///
    /// @brief Poll for completed operations from this Context.
    /// @param [in]     min_nr  Minimum number of events requested before returning.
    /// @param [in,out] nr      Input event capacity and output number of events returned.
    /// @param [out]    iocbp   Event output buffer.
    /// @param [in]     timeout Maximum amount of time to wait.
    ///
    void getStatus(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp,
                   struct timespec *timeout) override;

    ///
    /// @brief Cancel outstanding operations from this Context and wait for running ops.
    ///
    void cancelOperations() override;

private:
    const unsigned capacity;

    /// Per-Context mutex to limit access to one caller at a time.
    /// Shared as internally we can be more strategic about concurrent access.
    mutable std::shared_mutex context_mutex;

    /// Wakes callers waiting for operations to become terminal.
    std::condition_variable_any status_cv;

    /// An outstanding operation is a BatchOperation that has been submitted
    /// but is not yet complete or completed but not yet retrieved by the
    /// application.
    /// shared_ptr as it may need to be passed to a backend.
    std::unordered_set<std::shared_ptr<IBatchOperation>> outstanding_ops;

    /// Task group used for all submitted operations owned by this context.
    std::unique_ptr<ITaskGroup> task_group;

    BatchContext(unsigned capacity);

    friend class BatchContextMap;
};

class BatchContextMap {
public:
    /*!
     * @brief Create a new batch context
     * @param capacity Maximum number of outstanding operations that this context can manage
     * @return An opaque handle used to reference this new batch context
     */
    hipFileBatchHandle_t createContext(unsigned capacity);

    /*!
     * @brief Destroy a batch context and release all associated resources
     * @param handle The handle for the batch context to destroy
     */
    void destroyContext(hipFileBatchHandle_t handle);

    /*!
     * @brief Get a batch context
     * @param handle The opaque handle associated with a batch context
     * @return A batch context
     */
    std::shared_ptr<IBatchContext> get(hipFileBatchHandle_t handle);

    /*!
     * @brief Clear the contents
     */
    void clear();

private:
    /// batch context lookup table
    std::unordered_map<hipFileBatchHandle_t, std::shared_ptr<IBatchContext>> active_contexts;

    /// Mutex to protect the active context map
    mutable std::shared_mutex batch_mutex;
};

}
