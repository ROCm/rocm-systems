/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/

#pragma once

#include <ddDefs.h>

#define GPUOPEN_INTERFACE_MAJOR_VERSION 42

#define GPUOPEN_INTERFACE_MINOR_VERSION 1

#define GPUOPEN_INTERFACE_VERSION ((GPUOPEN_INTERFACE_MAJOR_VERSION << 16) | GPUOPEN_INTERFACE_MINOR_VERSION)

#define GPUOPEN_MINIMUM_INTERFACE_MAJOR_VERSION 38

#ifndef GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION
    static_assert(false, "Client must define GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION.");
#else
    static_assert((GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION >= GPUOPEN_MINIMUM_INTERFACE_MAJOR_VERSION) &&
        (GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION <= GPUOPEN_INTERFACE_MAJOR_VERSION),
        "The specified GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION is not supported.");
#endif

// Next version number for interface breaking changes
#define DD_UNRELEASED_MAJOR_VERSION 40

/*
***********************************************************************************************************************
*| Version | Change Description                                                                                       |
*| ------- | ---------------------------------------------------------------------------------------------------------|
*| 42.1    | Move Escape Commands to the shared header for access outside of message.h                                |
*| 42.0    | Updates RGP Protocol to support SPM counters and SE masking.                                             |
*| 41.0    | Updates DriverControlProtocol to allow user to query device clock frequencies for a given                |
*|         | clock mode without changing the clock mode.                                                              |
*| 40.0    | Moves DriverStatus enum out of DriverControlProtocol and into gpuopen.h, and renames several             |
*|         | DriverControlProtocol functions.                                                                         |
*| 39.0    | Simplified the LoggingClient interface to remove the internal pending message requirement.               |
*|         | Removed kInfiniteTimeout and replaced its uses with kLogicFailureTimeout.                                |
*|         | Decoupled RGP trace parameters from trace execution.                                                     |
*| 38.0    | Added support for specifying hostname in ListenerCreateInfo and renamed enableUWP flag to                |
*|         | enableKernelTransport.                                                                                   |
*| 37.0    | Added support for Querying ClientInfo from DriverControlProtocol                                         |
*| 36.1    | Removed internal log message queue inside LoggingClient. This improves performance significantly.        |
*| 36.0    | Added support for capturing the RGP trace on specific frame or dispatch.                                 |
*|         | Added bitfield to control whether driver internal code objects are included in the code object database. |
*| 35.0    | Updated Settings URI enum SettingType to avoid X11 macro name collision.                                 |
*| 34.0    | Updated URI services to define a version number for each service.                                        |
*| 33.0    | Abstracts URIRequestContext into an abstract interface.                                                  |
*| 32.0    | Updated RGPClient::EndTrace to support user specified timeout values. This allows tools to support       |
*|         | long running traces via user controlled cancellation dialogs.                                            |
*| 31.0    | Clean up DevDriverClient and DevDriverServer create info structs. Replace TransportCreateInfo            |
*|         | struct with MessageChannelCreateInfo and HostInfo structs.                                               |
*| 30.2    | Added support for RGP v6 protocol which supports trace trigger markers.                                  |
*| 30.1    | Add Push transfer support to the transfer protocol. Added PushBlock class, added v2 of the               |
*|         | TransferProtocol, and did a lot of internal cleanup. Legacy interfaces will be deprecated in a future    |
*|         | interface version change alongside URI changes.                                                          |
*| 30.0    | Remove CloseSession and OrphanSession from the public ISession object interface, and move the            |
*|         | functionality into the Session class.                                                                    |
*| 29.0    | Added a ResponseDataFormat enum to the URI protocol to distinguish between binary and text responses.    |
*| 28.0    | Formally deprecate legacy KMD client manager support in the Listener.                                    |
*| 27.2    | Updated FindFirstClient to support returning the matching ClientMetadata struct.                         |
*| 27.1    | Added PipelineDumpsEnabled status flag.                                                                  |
*| 27.0    | Deprecate global client status flags + replace it with client metadata.                                  |
*| 26.0    | Add new trace parameters in the RGP protocol.                                                            |
*| 25.0    | Initial refactor of LoggingProtocol. Removes Subcategories, being able to set/clear filter outside of a  |
*|         | trace, and significantly reduces the complexity that is involved in using it.                            |
*| 24.0    | Expanded driver initialization concept in driver control protocol.                                       |
*| 23.0    | Modified RGP client API usage pattern to be uniform across protocol versions.                            |
*| 22.0    | Refactor RGP client interface to support calculating transfer progress.                                  |
*| 21.1    | Added backwards compatible workaround for the session termination bug until we get the fix in mainline.  |
*| 21.0    | Enable link disconnection detection for socket based transports.                                         |
*| 20.0    | Added support for specifying the clock mode used during RGP profiling.                                   |
*| 19.0    | Refactor platform thread functions so that they are contained in a class. This is the last part of the   |
*|         | platform library that needed to be refactored, so future work will be focused on migrating the message   |
*|         | bus components to use the main platform library again.                                                   |
*| 18.0    | Consolidate DevDriver::DebugLevel and DevDriver::Message::DebugLevel into DevDriver::LogLevel.           |
*| 17.0    | Rename DD_VERSION_IS_SUPPORTED macro to DD_VERSION_SUPPORTS for conciseness.                             |
*| 16.1    | Rework session version negotiation to allow clients to support multiple server versions.                 |
*| 16.0    | Change semantics of CreateProtocolClient to AcquireProtocolClient. This aligns better with               |
*|         | ReleaseProtoclClient in terms of semantics.                                                              |
*| 15.0    | Added support for memory allocator callbacks via AllocCb.                                                |
*| 14.1    | Added DisableTrace call in RGP server to allow drivers to disable future traces if necessary.            |
*| 14.0    | Add TraceParameters to the RGP protocol to allow for configuration of trace behavior.                    |
*| 13.0    | Deprecate DevDriverClient::CreateProtocolClient() in favor of typesafe templated version.                |
*| 12.0    | Deprecate API features tied to legacy network protocol versions:                                         |
*|         | * Replace ConnectToRemoteClient with Connect, which now returns more detailed errors on failure          |
*|         | * Eliminate Send and ReceiveSessionMessage functions in IMsgChannel and SessionManager                   |
*|         | * Rename AuthenticationFailed to VersionMismatch since it is more semantically accurate                  |
*|         | * Update IMsgChannel::Update so that it takes a default timeout value, get rid of m_receiveTimeoutInMs   |
*| 11.5    | Updated server to remove GetVersion() call and pass version into AcceptSession() instead. This allows    |
*|         | servers to potentially implement backwards compatibility for older client versions. Additionally,        |
*|         | completely eliminate SessionTermination type in favor of expanding Result type. This allows propagating  |
*|         | more information on connection failures back to clients, as well as streamlines some code.               |
*| 11.4    | Implement per-protocol versioning. Client protocol is sent as part of session request, server decides    |
*|         | whether or not to accept session both from client and from version. Also rearrange how certain network   |
*|         | operations work: Syn now stores the initial session ID in the sessionId field of the message, Rst now    |
*|         | includes a result code, and closing a session now implicitly flushes both the client/server. Rst         |
*|         | Is also sent on just about every unknown session packet received, allowing faster error detection and    |
*|         | recovery. Bump network version number                                                                    |
*| 11.3    | Change ProcessId type from 64bit to 32bit integer and bump network protocol version.                     |
*| 11.2    | Update the network protocol to give external protocols values from 0-223 and system protocols 224-255.   |
*|         | Also clean up + deprecate some of the constants associated with protocols.                               |
*| 11.1    | Force alignment of all network transmitted structs, as well as pad. This is a breaking change for the    |
*|         | network protocol, but is otherwise API compatible.                                                       |
*| 11.0    | Deprecate the Protocol::ClientManangement enum, as well as ReadMessageBuffer and SendMessageBuffer in    |
*|         | message.lib.                                                                                             |
*| 10.0    | Remove callback from MessageChannel to prevent usage that can cause deadlocking.                         |
*| 9.0     | Formalized support for selective discard of non-session messages based on right in the message.          |
*|         | Implementation is that the sequence field of a message can be populated with the contents of a           |
*|         | ClientMetadata struct, which is then used by the receiving message channel to determine if it should     |
*|         | respond. Decision is based on whether or not the metadata matches the metadata of the receiving client.  |
*| 8.0     | Added support for default settings values in the settings protocol. Removed support for min and max      |
*|         | settings values since the scripts don't actually support those anyways.                                  |
*| 7.0     | Added a Finalize function to DevDriverServer and all protocol server objects. This function now handles  |
*|         | the wait on start functionality for drivers internally. Finalize should now be called instead of the old |
*|         | wait on start logic in client drivers.                                                                   |
*| 6.0     | Update client protocol management so that DevDriverClient no longer caches a single instance of each     |
*|         | client protocol, and add ability for clients to directly create more than one client protocol instance.  |
*|         | Additionally, make changes to underlying message channel/transport API that is not backwards compatible, |
*|         | as well as rename QueryClientInfoResponse to ClientInfoResponse and QueryClientInfoResponsePayload to    |
*|         | ClientInfoStruct.                                                                                        |
*| 5.0     | Update network protocol to allow specifying status flags at registration time, and add system message.   |
*|         | to indicate when a driver has been halted. Additionally, this changes the format of the client           |
*|         | registration packets so as to better detect version mismatch. It also fixes the ClientManangement typo.  |
*| 4.0     | Refactor interface so as to better delineate between system protcols/client protocols, as well as add    |
*|         | ability to query protocol availability. Requires version bump, so also formally deprecated               |
*|         | Result::Timeout and ClientStatusFlags::ProfilingEnabled, as well as moved entire SessionProtocol         |
*|         | namespace out of the public headers.                                                                     |
*| 3.1     | Introduce kNumberClientProtocols to replace usage of Protocol::Count                                     |
*| 3.0     | Rename SettingsProtocol::SettingType::Bool to Boolean to avoid conflict with Xlib macro. Additionally    |
*|         | formally deprecate Result::Timeout.                                                                      |
*| 2.2     | Added None (0) to ClientStatusFlags enum.                                                                |
*| 2.1     | Added kNamedPipeName to global namespace.                                                                |
*| 2.0     | Added functionality for enabling and disabling traces in RGPServer. Traces must now be explicitly        |
*|         | enabled before remote trace requests will succeed.                                                       |
*| 1.2     | Added AbortTrace() function to RGPServer.                                                                |
*| 1.1     | Added support for RGP protocol.                                                                          |
*| 1.0     | Initial versioned release.                                                                               |
***********************************************************************************************************************
*/

#define GPUOPEN_RGP_SPM_COUNTERS_VERSION                                      42
#define GPUOPEN_DRIVER_CONTROL_QUERY_CLOCKS_BY_MODE_VERSION                   41
#define GPUOPEN_DRIVER_CONTROL_CLEANUP_VERSION                                40
#define GPUOPEN_DECOUPLED_RGP_PARAMETERS_VERSION                              39
#define GPUOPEN_SIMPLER_LOGGING_VERSION                                       39
#define GPUOPEN_LISTENER_HOSTNAME_VERSION                                     38
#define GPUOPEN_SETTINGS_URI_LINUX_BUILD                                      35
#define GPUOPEN_VERSIONED_URI_SERVICES_VERSION                                34
#define GPUOPEN_URIINTERFACE_CLEANUP_VERSION                                  33
#define GPUOPEN_LONG_RGP_TRACES_VERSION                                       32
#define GPUOPEN_CREATE_INFO_CLEANUP_VERSION                                   31
#define GPUOPEN_SESSION_INTERFACE_CLEANUP_VERSION                             30
#define GPUOPEN_URI_RESPONSE_FORMATS_VERSION                                  29
#define GPUOPEN_DEPRECATE_LEGACY_KMD_VERSION                                  28
#define GPUOPEN_DISTRIBUTED_STATUS_FLAGS_VERSION                              27
#define GPUOPEN_RGP_TRACE_PARAMETERS_V3_VERSION                               26
#define GPUOPEN_LOGGING_SIMPLIFICATION_VERSION                                25
#define GPUOPEN_DRIVERCONTROL_INITIALIZATION_VERSION                          24
#define GPUOPEN_RGP_UNIFORM_API_VERSION                                       23
#define GPUOPEN_RGP_PROGRESS_VERSION                                          22
#define GPUOPEN_KEEPALIVE_VERSION                                             21
#define GPUOPEN_PROFILING_CLOCK_MODES_VERSION                                 20
#define GPUOPEN_THREAD_REFACTOR_VERSION                                       19
#define GPUOPEN_LOGLEVEL_CLEANUP_VERSION                                      18
#define GPUOPEN_RENAME_MACRO_VERSION                                          17
#define GPUOPEN_PROTOCOL_CLIENT_REUSE_VERSION                                 16
#define GPUOPEN_MEMORY_ALLOCATORS_VERSION                                     15
#define GPUOPEN_RGP_TRACE_PARAMETERS_VERSION                                  14
#define GPUOPEN_DEPRECATE_CREATEPROTOCOLCLIENT_VERSION                        13
#define GPUOPEN_DEPRECATE_LEGACY_NETAPI_VERSION                               12
#define GPUOPEN_POST_GDC_CLEANUP_VERSION                                      11
#define GPUOPEN_DEPRECATE_EXTERNAL_CALLBACK_VERSION                           10
#define GPUOPEN_SELECTIVE_RESPOND_VERSION                                      9
#define GPUOPEN_DEFAULT_SETTINGS_VERSION                                       8
#define GPUOPEN_SERVER_FINALIZE_VERSION                                        7
#define GPUOPEN_DEPRECATE_LEGACY_VERSION                                       6
#define GPUOPEN_CLIENT_REGISTRATION_VERSION                                    5
#define GPUOPEN_PROTOCOL_CLEANUP_VERSION                                       4
#define GPUOPEN_LINUX_BUILD_VERSION                                            3
#define GPUOPEN_EXPLICIT_ENABLE_RGP_VERSION                                    2
#define GPUOPEN_INITIAL_VERSION                                                1

// This will be properly defined when RMV 1.1 features are complete, defining it now allows
// clients to code to the interface ahead of all of the work being complete.
#define GPUOPEN_RMV_1_1_VERSION 0xFFFF

#define DD_VERSION_SUPPORTS(x) (GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION >= x)

namespace DevDriver
{
    typedef uint16_t ClientId;
    typedef uint32_t SessionId;
    typedef uint8_t  MessageCode;
    typedef uint16_t WindowSize;
    typedef uint64_t Sequence;
    typedef uint16_t Version;
    typedef uint16_t StatusFlags;

#if DD_VERSION_SUPPORTS(GPUOPEN_SIMPLER_LOGGING_VERSION)
    // A common timeout in milliseconds for components to use when they do not expect timeout to fail.
    // If an operation that uses this timeout returns Result::NotReady, consider it a fatal error.
    DD_STATIC_CONST uint32 kLogicFailureTimeout = 1000;
#else
    DD_STATIC_CONST uint32 kInfiniteTimeout = ~(0u);
    DD_STATIC_CONST uint32 kLogicFailureTimeout = kInfiniteTimeout;
#endif
    DD_STATIC_CONST uint32 kNoWait = (0u);

    ////////////////////////////
    // Driver states
    enum struct DriverStatus : uint32
    {
        Running = 0,
        Paused,
        HaltedOnDeviceInit,
        EarlyDeviceInit,
        LateDeviceInit,
        PlatformInit,
        HaltedOnPlatformInit,
        HaltedPostDeviceInit,
        Count
    };

    ////////////////////////////
    // Client status codes
    enum struct ClientStatusFlags : StatusFlags
    {
        None                   = 0,
        DeveloperModeEnabled   = (1 << 0),
        DeviceHaltOnConnect    = (1 << 1),
        GpuCrashDumpsEnabled   = (1 << 2),
        PipelineDumpsEnabled   = (1 << 3),
        PlatformHaltOnConnect  = (1 << 4),
        DriverInitializer      = (1 << 5)
    };

    DD_CHECK_SIZE(ClientId, 2);
    DD_STATIC_CONST int16 kRouterPrefixWidth = 3;
    DD_STATIC_CONST int16 kRouterPrefixShift = (int16)(16 - kRouterPrefixWidth);
    DD_STATIC_CONST ClientId kClientIdMask = (1 << kRouterPrefixShift) - 1;
    DD_STATIC_CONST ClientId kRouterPrefixMask = static_cast<ClientId>(~(kClientIdMask));

    union ProtocolFlags
    {
        struct DD_ALIGNAS(4)
        {
            // TODO: Replace logging, settings, and gpuCrashDump with "reserved" once all driver usage is removed.
            uint32 logging          : 1;
            uint32 settings         : 1;
            uint32 driverControl    : 1;
            uint32 rgp              : 1;
            uint32 etw              : 1;
            uint32 gpuCrashDump     : 1;
            uint32 event            : 1;
            uint32 reserved         : 25;
        };
        uint32 value;
    };

    DD_CHECK_SIZE(ProtocolFlags, 4);

    ////////////////////////////
    // Component definitions
    enum struct Component : uint8
    {
        Unknown = 0,
        Server,
        Tool,
        Driver,
        Count
    };

    struct DD_ALIGNAS(4) ClientMetadata
    {
        ProtocolFlags protocols;
        Component     clientType;
        uint8         reserved;
        StatusFlags   status;

        // For System messages, which are not session-based, we alias the sequence field as ClientMetadata.  This constructor
        // is provided to help unpack the raw 64-bit sequence field into a ClientMetadata struct without needing to type-cast
        explicit ClientMetadata(uint64 value)
        {
            // If we're going to alias as a 64-bit value, make sure the struct is still just 64-bits)
            static_assert(sizeof(uint64) == sizeof(ClientMetadata),
                          "Size of ClientMetadata is no longer 64-bits, alias constructor needs updating");

            // Bits 0-31 are the ProtocolFlags
            protocols.value = static_cast<uint32>(value & 0xFFFF);

            // Bits 32-39 are the Component
            clientType = static_cast<Component>((value & 0xFF00000000) >> 32);

            // Bits 40-47 are reserved, ignore them and zero initialize
            reserved = 0;

            // Bits 48-63 are the StatusFlags
            status = static_cast<StatusFlags>((value & 0xFFFF000000000000) >> 48);
        }

        // Default constructor, default initialize everything
        ClientMetadata() = default;

        // Returns true if all values are default values
        bool IsDefault() const
        {
            return ((protocols.value == 0) && (clientType == Component::Unknown) && (status == 0));
        }

        // Test if all non-zero fields in the ClientMetadata value are contained in the function parameter
        bool Matches(const ClientMetadata &right) const
        {
            bool result = true;

            // The Matches function treats this struct as a filter, so a ClientMetadata with all default (zero) values
            // by definition always matches.
            if (IsDefault() == false)
            {
                // Component is an enum, so the comparison needs to be equality
                const bool clientTypeMatches =
                    (clientType != Component::Unknown)
                    ? (clientType == right.clientType)
                    : true;

                // ProtocolFlags is a bit field, so we can do a bitwise comparison
                const bool protocolMatches =
                    (protocols.value != 0)
                    ? (protocols.value & right.protocols.value) == protocols.value
                    : true;
                // StatusFlags is a bit field, so we can do a bitwise comparison
                const bool statusMatches =
                    (status != 0)
                    ? (status & right.status) == status
                    : true;
                result = clientTypeMatches & protocolMatches & statusMatches;
            }

            return result;
        }

        // Test if any non-zero fields in the ClientMetadata value are contained in the function parameter
        bool MatchesAny(const ClientMetadata &right) const
        {
            bool result = true;

            // The MatchesAny function treats this struct as a filter, so a ClientMetadata with all default (zero) values
            // by definition always matches.
            if (IsDefault() == false)
            {
                // Component is an enum, so the comparison needs to be equality
                const bool clientTypeMatches = (clientType == right.clientType);
                // ProtocolFlags is a bit field, so we can do a bitwise comparison
                const bool protocolMatches = (protocols.value & right.protocols.value) != 0;
                // StatusFlags is a bit field, so we can do a bitwise comparison
                const bool statusMatches = (status & right.status) != 0;
                result = clientTypeMatches | protocolMatches | statusMatches;
            }

            return result;
        }
    };

    DD_CHECK_SIZE(ClientMetadata, 8);

    ////////////////////////////
    // Protocol definitions
    enum struct Protocol : uint8
    {
        DriverControl = 0,
        Reserved0,
        Reserved1,
        RGP,
        ETW,
        Reserved2,
        Event,
        DefinedProtocolCount,

        // System enumerations
        MaxUserProtocol = 223,
        /* RESERVED FOR SYSTEM USE */
        Transfer = 251,
        URI = 252,
        Session = 253,
        ClientManagement = 254,
        System = 255,
    };

    // this gives you the number of pre-defined user protocols that exist
    DD_STATIC_CONST uint32 kNumberClientProtocols = static_cast<uint32>(Protocol::DefinedProtocolCount);

    // this gives you the maximum number of client protocols you can reserve.
    DD_STATIC_CONST uint32 kMaxClientProtocolId = static_cast<uint32>(Protocol::MaxUserProtocol);

    static_assert(kNumberClientProtocols <= (kMaxClientProtocolId + 1), "Invalid protocol definitions specified");

    ///////////////////////
    // General definitions
    DD_STATIC_CONST uint32 kMessageVersion = 1011;

    // Max string size for names and messages
    DD_STATIC_CONST Size kMaxStringLength = 128;

    // Broadcast client ID
    DD_STATIC_CONST ClientId kBroadcastClientId = 0;

    // Invalid Session ID
    DD_STATIC_CONST SessionId kInvalidSessionId = 0;

    // Default network port number
    DD_STATIC_CONST uint16_t kDefaultNetworkPort = 27300;

    // Transport type enumeration
    enum class TransportType : uint32
    {
        Local = 0,
        Remote,
#if defined(DD_PLATFORM_WINDOWS_UM)
        MessageBus,
#endif
    };

    // Struct used to designate a transport type, port number, and hostname
    struct HostInfo
    {
        TransportType type;      // Transport type, as defined above
        uint16_t      port;      // Port number if applicable
        const char*   pHostname; // Host address, address, or path
    };

    // Default local host information
    DD_STATIC_CONST HostInfo kDefaultLocalHost =
    {
        TransportType::Remote,
        kDefaultNetworkPort,
        "localhost"
    };

    // Default named pipe information
    DD_STATIC_CONST HostInfo kDefaultNamedPipe =
    {
        TransportType::Local,
        0,
        nullptr
    };

#if defined(DD_PLATFORM_WINDOWS_UM)
    // Default message bus information
    DD_STATIC_CONST HostInfo kMessageBus =
    {
        TransportType::MessageBus,
        0,
        nullptr
    };
#endif

    ////////////////////////////
    // Common definition of a message header
    //
    // todo: better packing of these values
    //       - payloadSize needs to be moved to where windowSize is currently
    //       - windowSize, sessionId, and sequence need to be moved into protocol specific payloads
    //       - minimum alignment could then be reduced to 2 bytes, and min packet size would be 8 bytes
    //       - downside is that pretty much every protocol would need to define some extra data

    DD_NETWORK_STRUCT(MessageHeader, 8)
    {
        // source and destination client ids
        ClientId            srcClientId;    //   0 -  15
        ClientId            dstClientId;    //  16 -  31

        // protocol and command
        Protocol            protocolId;     //  31 -  38
        MessageCode         messageId;      //  39 -  47
        WindowSize          windowSize;     //  48 -  63

        // payload size + current session ID
        Size                payloadSize;    //  64 -  91
        SessionId           sessionId;      //  92 - 127

        // sequence number when using a session
        Sequence            sequence;       // 128 - 191
    };

    DD_CHECK_SIZE(MessageHeader, 24);

    DD_STATIC_CONST Size kMaxMessageSizeInBytes = 1408;
    DD_STATIC_CONST Size kMaxPayloadSizeInBytes = (kMaxMessageSizeInBytes - sizeof(MessageHeader));

    DD_NETWORK_STRUCT(MessageBuffer, 8)
    {
        MessageHeader   header;
        char            payload[kMaxPayloadSizeInBytes];
    };

    DD_CHECK_SIZE(MessageBuffer, sizeof(MessageHeader) + kMaxPayloadSizeInBytes);

    // Helper function used to validate message buffers that arrive from an external source
    // Returns Success if the message buffer is valid and Error otherwise.
    inline Result ValidateMessageBuffer(const void* pMsgBuffer, size_t msgBufferSize)
    {
        Result result = Result::Error;

        // Ensure that we've been passed valid parameters
        if ((pMsgBuffer != nullptr) && (msgBufferSize > 0))
        {
            // A valid message buffer must be no larger than the full size message buffer structure
            // and it must also be large enough to contain a valid header.
            if ((msgBufferSize <= sizeof(MessageBuffer)) && (msgBufferSize >= sizeof(MessageHeader)))
            {
                // Calculate the total size of the message from the data encoded in the buffer.
                const MessageHeader* pHeader = reinterpret_cast<const MessageHeader*>(pMsgBuffer);
                const size_t encodedMessageSize = (sizeof(MessageHeader) + pHeader->payloadSize);

                // The encoded message size should match our expected size exactly
                if (encodedMessageSize == msgBufferSize)
                {
                    result = Result::Success;
                }
            }
        }
        else
        {
            result = Result::InvalidParameter;
        }

        return result;
    }

    // tripwire - this intentionally will break if the message version changes. Since these are breaking changes already, we need to address
    // this problem when it happens.
    static_assert(kMessageVersion == 1011, "ClientInfoStruct needs to be updated so that clientName is long enough to support a full path");
    // todo: shorten clientDescription to 64bytes and make clientName 320bytes to support full path
    DD_NETWORK_STRUCT(ClientInfoStruct, 4)
    {
        char            clientName[kMaxStringLength];
        char            clientDescription[kMaxStringLength];
        // reserve 128bytes in case we need another string in the future
        char            reserved[kMaxStringLength];
        ClientMetadata  metadata;
        ProcessId       processId;
        // pad this out to 512 bytes for future expansion
        char            padding[116];
    };

    DD_CHECK_SIZE(ClientInfoStruct, 512);

    ///////////////////////
    // GPU Open Message codes
    enum struct EscapeCommand : uint32
    {
        Unknown = 0,
        QueryStatus,              // Will be deprecated in a future change
        RegisterClient,
        UnregisterClient,
        RegisterExternalClient,   // Will be deprecated in a future change
        UnregisterExternalClient, // Will be deprecated in a future change
        UpdateClientStatus,       // Will be deprecated in a future change
        QueryCapabilities,
        EnableDeveloperMode,
        DisableDeveloperMode,
        QueryDeveloperModeStatus,
        RegisterRouter,
        UnregisterRouter,
        AmdLogEvent,
        Count
    };
}
