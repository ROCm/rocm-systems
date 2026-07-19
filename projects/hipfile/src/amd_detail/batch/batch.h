/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}

namespace hipFile {

struct InvalidBatchHandle : public std::invalid_argument {
    InvalidBatchHandle() : std::invalid_argument{"Invalid batch handle"}
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
        // Every state must also define:
        //     static constexpr bool isTerminal() noexcept;
        // which is true when the operation has reached a final state and will
        // never run again. It is static so that it can be queried on a state
        // type that is not default constructible.

        ssize_t ret() const noexcept
        {
            return 0;
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

        static constexpr bool isTerminal() noexcept
        {
            return false;
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

        static constexpr bool isTerminal() noexcept
        {
            return false;
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

        static constexpr bool isTerminal() noexcept
        {
            return false;
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

        static constexpr bool isTerminal() noexcept
        {
            return true;
        }

        ssize_t ret() const noexcept
        {
            return num_bytes;
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

        static constexpr bool isTerminal() noexcept
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

        static constexpr bool isTerminal() noexcept
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

        static constexpr bool isTerminal() noexcept
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

        static constexpr bool isTerminal() noexcept
        {
            return true;
        }

        ssize_t ret() const noexcept
        {
            return error;
        }

        Failed transitionTo(const Failed &next) const;

        ssize_t error;
    };

    using OperationState =
        std::variant<Waiting, Pending, Running, Complete, Canceled, Invalid, Timeout, Failed>;
}

/// @brief Represents a single IO Request
class BatchOperation {
public:
    // Don't allow copying
    BatchOperation(const BatchOperation &)            = delete;
    BatchOperation &operator=(const BatchOperation &) = delete;

    // Don't allow moving
    BatchOperation(BatchOperation &&)            = delete;
    BatchOperation &operator=(BatchOperation &&) = delete;

    /// @brief Create an operation to handle and track an IO request.
    /// @param [in] params IO parameters
    /// @param [in] buffer Buffer corresponding to params->u.batch.devPtr_base
    /// @param [in] file File corresponding params->fh
    BatchOperation(std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                   std::shared_ptr<IFile> file);

    /// @brief Mark the operation as accepted and ready to run.
    void markPending();

    /// @brief Cancel the operation if it can be transitioned to Canceled; otherwise no-op.
    /// @return True if the operation now has Canceled state, false otherwise.
    bool tryCancel();

    /// @brief Record an internal execution failure on the operation.
    void recordInternalError();

    /// @brief Return a snapshot of the operation event state.
    hipFileIOEvents_t event() const;

private:
    /// @brief A copy of the params provided by the application.
    /// @internal Keep this listed at the top of BatchOperation.
    const std::unique_ptr<const hipFileIOParams_t> io_params;

    /// @brief A reference to the specified Buffer.
    std::shared_ptr<IBuffer> buffer;

    /// @brief A reference to the specified registered File.
    std::shared_ptr<IFile> file;

    /// @brief Protects operation state.
    mutable std::mutex state_mutex;

    /// @brief Current operation state.
    batchOperationState::OperationState state{batchOperationState::Waiting{}};

    /// @brief Move to the next operation state. Caller must hold state_mutex.
    template <class Next> void transitionTo(Next next);
};

using BatchOperations = std::vector<std::shared_ptr<BatchOperation>>;

class IBatchContext {
public:
    static constexpr unsigned MAX_SIZE = 128;

    virtual ~IBatchContext()                               = default;
    virtual unsigned getCapacity() const noexcept          = 0;
    virtual void     submitOperations(BatchOperations ops) = 0;
};

class BatchContext : public IBatchContext {
public:
    // Don't allow copying
    BatchContext(const BatchContext &)            = delete;
    BatchContext &operator=(const BatchContext &) = delete;

    // Don't allow moving
    BatchContext(BatchContext &&)            = delete;
    BatchContext &operator=(BatchContext &&) = delete;

    ///
    /// @brief Return the max number of concurrent operations supported by this BatchContext.
    ///
    /// @return The max number of concurrent operations that can be processed by this BatchContext.
    /// @note This may not exceed the value returned by `MAX_SIZE`.
    unsigned getCapacity() const noexcept override;

    ///
    /// @brief Submit one or more operations to this Context.
    /// @param [in] ops Operations to enqueue.
    ///
    /// @note This is an All or None operation.
    ///
    void submitOperations(BatchOperations ops) override;

private:
    const unsigned capacity;

    /// Per-Context mutex to limit access to one caller at a time.
    /// Shared as internally we can be more strategic about concurrent access.
    mutable std::shared_mutex context_mutex;

    /// An outstanding operation is a BatchOperation that has been submitted
    /// but is not yet complete or completed but not yet retrieved by the
    /// application.
    /// shared_ptr as it may need to be passed to a backend.
    std::unordered_set<std::shared_ptr<BatchOperation>> outstanding_ops;

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
