/*
 * Test suite for libhsakmt built with the amdgpu_lite backend.
 *
 * Ports applicable tests from kfdtest (KFDOpenCloseKFDTest,
 * KFDTopologyTest) that exercise the APIs we implement:
 *   - open/close with ref counting
 *   - version query
 *   - topology: system properties, node properties, memory
 *     properties, cache properties, IO link properties
 *   - parameter validation (NULL pointers, invalid node IDs)
 *   - fork isolation (child process without open)
 *   - stub verification (unimplemented APIs return NOT_SUPPORTED)
 *
 * Requires /dev/amdgpu_lite0 (amdgpu_lite.ko loaded with GPU).
 *
 * Build:
 *   gcc -o amdgpu_lite_test main.c -I../../include -L../../build-lite \
 *       -lhsakmt -lpthread -lrt -ldl
 *
 * Run:
 *   sudo ./amdgpu_lite_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <hsakmt/hsakmt.h>

static int pass_count = 0;
static int fail_count = 0;

#define TEST(name) do { printf("  %-50s ", name); } while (0)
#define PASS() do { printf("PASS\n"); pass_count++; } while (0)
#define PASS_FMT(fmt, ...) do { printf("PASS (" fmt ")\n", ##__VA_ARGS__); pass_count++; } while (0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fail_count++; } while (0)

/* ======================================================================
 * KFDCloseKFDTest::CloseAClosedKfd
 * Verify closing without opening returns proper error.
 * ====================================================================== */
static void test_close_a_closed_kfd(void)
{
	TEST("CloseAClosedKfd");
	if (hsaKmtCloseKFD() == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED) {
		PASS();
	} else {
		FAIL("expected KERNEL_IO_CHANNEL_NOT_OPENED");
	}
}

/* ======================================================================
 * KFDOpenCloseKFDTest::OpenCloseKFD
 * Basic open/close cycle.
 * ====================================================================== */
static void test_open_close_kfd(void)
{
	TEST("OpenCloseKFD");
	HSAKMT_STATUS s = hsaKmtOpenKFD();
	if (s != HSAKMT_STATUS_SUCCESS) {
		FAIL("open failed");
		printf("    (Is /dev/amdgpu_lite0 present? Is amdgpu_lite.ko loaded?)\n");
		exit(1);
	}
	s = hsaKmtCloseKFD();
	if (s == HSAKMT_STATUS_SUCCESS) {
		PASS();
	} else {
		FAIL("close failed");
	}
}

/* ======================================================================
 * KFDOpenCloseKFDTest::OpenAlreadyOpenedKFD
 * Opening twice returns KERNEL_ALREADY_OPENED; close both.
 * ====================================================================== */
static void test_open_already_opened(void)
{
	HSAKMT_STATUS s;

	s = hsaKmtOpenKFD();
	if (s != HSAKMT_STATUS_SUCCESS) {
		printf("  FATAL: first open failed\n");
		exit(1);
	}

	TEST("OpenAlreadyOpenedKFD");
	s = hsaKmtOpenKFD();
	if (s == HSAKMT_STATUS_KERNEL_ALREADY_OPENED) {
		PASS();
	} else {
		FAIL("expected KERNEL_ALREADY_OPENED");
	}

	/* Close both opens */
	hsaKmtCloseKFD();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDOpenCloseKFDTest::InvalidKFDHandleTest
 * Fork child: GetVersion without open must fail.
 * ====================================================================== */
static void test_invalid_kfd_handle(void)
{
	TEST("InvalidKFDHandleTest (fork)");

	HSAKMT_STATUS s = hsaKmtOpenKFD();
	if (s != HSAKMT_STATUS_SUCCESS) {
		FAIL("parent open failed");
		return;
	}

	pid_t child = fork();
	if (child == 0) {
		/* Child: KFD is NOT open in this process context */
		HsaVersionInfo ver;
		HSAKMT_STATUS cs = hsaKmtGetVersion(&ver);
		/* Should fail because child never called OpenKFD */
		exit(cs == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED ? 0 : 1);
	} else if (child > 0) {
		int status;
		waitpid(child, &status, 0);
		if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
			PASS();
		} else {
			FAIL("child did not get KERNEL_IO_CHANNEL_NOT_OPENED");
		}
	} else {
		FAIL("fork() failed");
	}

	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::BasicTest
 * Validate node properties for all nodes.
 * ====================================================================== */
static void test_topology_basic(void)
{
	HSAKMT_STATUS s;
	HsaSystemProperties sys_props;
	HsaNodeProperties props;
	unsigned int node;

	s = hsaKmtOpenKFD();
	if (s != HSAKMT_STATUS_SUCCESS) {
		printf("  FATAL: open failed\n");
		exit(1);
	}

	memset(&sys_props, 0, sizeof(sys_props));
	s = hsaKmtAcquireSystemProperties(&sys_props);
	if (s != HSAKMT_STATUS_SUCCESS || sys_props.NumNodes == 0) {
		printf("  FATAL: AcquireSystemProperties failed\n");
		exit(1);
	}

	for (node = 0; node < sys_props.NumNodes; node++) {
		memset(&props, 0, sizeof(props));
		s = hsaKmtGetNodeProperties(node, &props);
		if (s != HSAKMT_STATUS_SUCCESS)
			continue;

		if (props.DeviceId == 0) {
			/* CPU-only node */
			char name[64];
			snprintf(name, sizeof(name),
				 "BasicTest: CPU node %u has cores", node);
			TEST(name);
			if (props.NumCPUCores > 0) {
				PASS_FMT("%u cores", props.NumCPUCores);
			} else {
				FAIL("NumCPUCores == 0");
			}
		} else {
			/* GPU node */
			char name[64];

			snprintf(name, sizeof(name),
				 "BasicTest: GPU node %u has compute cores",
				 node);
			TEST(name);
			if (props.NumFComputeCores > 0) {
				PASS_FMT("%u", props.NumFComputeCores);
			} else {
				FAIL("NumFComputeCores == 0");
			}

			snprintf(name, sizeof(name),
				 "BasicTest: GPU node %u uCode > 0", node);
			TEST(name);
			if (props.EngineId.ui32.uCode > 0) {
				PASS();
			} else {
				FAIL("uCode == 0");
			}

			snprintf(name, sizeof(name),
				 "BasicTest: GPU node %u Major >= 7", node);
			TEST(name);
			if (props.EngineId.ui32.Major >= 7) {
				PASS_FMT("Major=%u", props.EngineId.ui32.Major);
			} else {
				FAIL("Major < 7");
			}

			snprintf(name, sizeof(name),
				 "BasicTest: GPU node %u Minor < 10", node);
			TEST(name);
			if (props.EngineId.ui32.Minor < 10) {
				PASS();
			} else {
				FAIL("Minor >= 10");
			}

			snprintf(name, sizeof(name),
				 "BasicTest: GPU node %u SDMA fw > 0", node);
			TEST(name);
			if (props.uCodeEngineVersions.uCodeSDMA > 0) {
				PASS();
			} else {
				FAIL("uCodeSDMA == 0");
			}

			snprintf(name, sizeof(name),
				 "BasicTest: GPU node %u VGPR/SGPR sizes",
				 node);
			TEST(name);
			if (props.VGPRSizePerCU > 0 && props.SGPRSizePerCU > 0) {
				PASS_FMT("VGPR=%uK SGPR=%uK",
					 props.VGPRSizePerCU / 1024,
					 props.SGPRSizePerCU / 1024);
			} else {
				FAIL("VGPR or SGPR size == 0");
			}
		}

		/* All nodes must have memory banks */
		{
			char name[64];
			snprintf(name, sizeof(name),
				 "BasicTest: node %u has memory banks", node);
			TEST(name);
			if (props.NumMemoryBanks > 0) {
				PASS_FMT("%u banks", props.NumMemoryBanks);
			} else {
				FAIL("NumMemoryBanks == 0");
			}
		}
	}

	hsaKmtReleaseSystemProperties();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodePropertiesInvalidParams
 * NULL pointer must return INVALID_PARAMETER.
 * ====================================================================== */
static void test_get_node_properties_null(void)
{
	hsaKmtOpenKFD();

	TEST("GetNodePropertiesInvalidParams (NULL)");
	if (hsaKmtGetNodeProperties(0, NULL) ==
	    HSAKMT_STATUS_INVALID_PARAMETER) {
		PASS();
	} else {
		FAIL("expected INVALID_PARAMETER");
	}

	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodePropertiesInvalidNodeNum
 * Out-of-range node must return INVALID_NODE_UNIT.
 * ====================================================================== */
static void test_get_node_properties_invalid_node(void)
{
	HsaSystemProperties sys_props;
	HsaNodeProperties props;

	hsaKmtOpenKFD();
	hsaKmtAcquireSystemProperties(&sys_props);

	char name[64];
	snprintf(name, sizeof(name),
		 "GetNodePropertiesInvalidNodeNum (node=%u)",
		 sys_props.NumNodes);
	TEST(name);

	if (hsaKmtGetNodeProperties(sys_props.NumNodes, &props) ==
	    HSAKMT_STATUS_INVALID_NODE_UNIT) {
		PASS();
	} else {
		FAIL("expected INVALID_NODE_UNIT");
	}

	hsaKmtReleaseSystemProperties();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodeMemoryProperties
 * Query memory properties for all nodes.
 * ====================================================================== */
static void test_get_node_memory_properties(void)
{
	HsaSystemProperties sys_props;
	HsaNodeProperties props;
	unsigned int node;

	hsaKmtOpenKFD();
	hsaKmtAcquireSystemProperties(&sys_props);

	for (node = 0; node < sys_props.NumNodes; node++) {
		hsaKmtGetNodeProperties(node, &props);
		if (props.NumMemoryBanks == 0)
			continue;

		HsaMemoryProperties *mem = calloc(props.NumMemoryBanks,
						  sizeof(*mem));
		char name[64];
		snprintf(name, sizeof(name),
			 "GetNodeMemoryProperties (node %u, %u banks)",
			 node, props.NumMemoryBanks);
		TEST(name);

		HSAKMT_STATUS s = hsaKmtGetNodeMemoryProperties(
			node, props.NumMemoryBanks, mem);
		if (s == HSAKMT_STATUS_SUCCESS) {
			PASS();
		} else {
			FAIL("query failed");
		}
		free(mem);
	}

	hsaKmtReleaseSystemProperties();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GpuvmApertureValidate
 * GPU nodes must have a frame buffer heap (PRIVATE or PUBLIC).
 * ====================================================================== */
static void test_gpuvm_aperture_validate(void)
{
	HsaSystemProperties sys_props;
	HsaNodeProperties props;
	unsigned int node;

	hsaKmtOpenKFD();
	hsaKmtAcquireSystemProperties(&sys_props);

	for (node = 0; node < sys_props.NumNodes; node++) {
		hsaKmtGetNodeProperties(node, &props);
		if (props.DeviceId == 0)
			continue; /* skip CPU nodes */

		HsaMemoryProperties *mem = calloc(props.NumMemoryBanks,
						  sizeof(*mem));
		hsaKmtGetNodeMemoryProperties(node, props.NumMemoryBanks, mem);

		int found = 0;
		unsigned int bank;
		for (bank = 0; bank < props.NumMemoryBanks; bank++) {
			if (mem[bank].HeapType ==
				    HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE ||
			    mem[bank].HeapType ==
				    HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
				found = 1;
		}

		char name[64];
		snprintf(name, sizeof(name),
			 "GpuvmApertureValidate (GPU node %u)", node);
		TEST(name);
		if (found) {
			PASS();
		} else {
			FAIL("no frame buffer heap found");
		}
		free(mem);
	}

	hsaKmtReleaseSystemProperties();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodeCacheProperties
 * Query cache properties for all nodes (may be zero).
 * ====================================================================== */
static void test_get_node_cache_properties(void)
{
	HsaSystemProperties sys_props;
	HsaNodeProperties props;
	unsigned int node;

	hsaKmtOpenKFD();
	hsaKmtAcquireSystemProperties(&sys_props);

	for (node = 0; node < sys_props.NumNodes; node++) {
		hsaKmtGetNodeProperties(node, &props);

		HsaCacheProperties *cache = NULL;
		if (props.NumCaches > 0)
			cache = calloc(props.NumCaches, sizeof(*cache));

		char name[64];
		snprintf(name, sizeof(name),
			 "GetNodeCacheProperties (node %u, %u caches)",
			 node, props.NumCaches);
		TEST(name);

		HSAKMT_STATUS s = hsaKmtGetNodeCacheProperties(
			node, props.CComputeIdLo, props.NumCaches, cache);
		if (s == HSAKMT_STATUS_SUCCESS) {
			PASS();
		} else {
			FAIL("query failed");
		}
		free(cache);
	}

	hsaKmtReleaseSystemProperties();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * KFDTopologyTest::GetNodeIoLinkProperties
 * Query IO link properties; validate NodeFrom matches query node.
 * ====================================================================== */
static void test_get_node_iolink_properties(void)
{
	HsaSystemProperties sys_props;
	HsaNodeProperties props;
	unsigned int node;

	hsaKmtOpenKFD();
	hsaKmtAcquireSystemProperties(&sys_props);

	for (node = 0; node < sys_props.NumNodes; node++) {
		hsaKmtGetNodeProperties(node, &props);
		if (props.NumIOLinks == 0)
			continue;

		HsaIoLinkProperties *links = calloc(props.NumIOLinks,
						    sizeof(*links));
		char name[64];
		snprintf(name, sizeof(name),
			 "GetNodeIoLinkProperties (node %u, %u links)",
			 node, props.NumIOLinks);
		TEST(name);

		HSAKMT_STATUS s = hsaKmtGetNodeIoLinkProperties(
			node, props.NumIOLinks, links);
		if (s != HSAKMT_STATUS_SUCCESS) {
			FAIL("query failed");
			free(links);
			continue;
		}

		/* First link's NodeFrom must match query node */
		if (links[0].NodeFrom == node) {
			PASS_FMT("[%u]--(%u)-->[%u]",
				 links[0].NodeFrom,
				 links[0].Weight,
				 links[0].NodeTo);
		} else {
			FAIL("NodeFrom mismatch");
		}
		free(links);
	}

	hsaKmtReleaseSystemProperties();
	hsaKmtCloseKFD();
}

/* ======================================================================
 * Stub verification
 * ====================================================================== */
static void test_stubs(void)
{
	HSAKMT_STATUS s;

	hsaKmtOpenKFD();

	TEST("hsaKmtAllocMemory (stub)");
	void *addr = NULL;
	HsaMemFlags flags;
	memset(&flags, 0, sizeof(flags));
	s = hsaKmtAllocMemory(1, 4096, flags, &addr);
	if (s == HSAKMT_STATUS_NOT_SUPPORTED) {
		PASS();
	} else {
		FAIL("expected NOT_SUPPORTED");
	}

	TEST("hsaKmtCreateQueue (stub)");
	HsaQueueResource qr;
	memset(&qr, 0, sizeof(qr));
	s = hsaKmtCreateQueue(1, HSA_QUEUE_COMPUTE, 100,
			      HSA_QUEUE_PRIORITY_NORMAL,
			      NULL, 4096, NULL, &qr);
	if (s == HSAKMT_STATUS_NOT_SUPPORTED) {
		PASS();
	} else {
		FAIL("expected NOT_SUPPORTED");
	}

	TEST("hsaKmtCreateEvent (stub)");
	HsaEvent *event = NULL;
	HsaEventDescriptor desc;
	memset(&desc, 0, sizeof(desc));
	s = hsaKmtCreateEvent(&desc, false, false, &event);
	if (s == HSAKMT_STATUS_NOT_SUPPORTED) {
		PASS();
	} else {
		FAIL("expected NOT_SUPPORTED");
	}

	TEST("hsaKmtSetMemoryPolicy (stub)");
	s = hsaKmtSetMemoryPolicy(0, 0, 0, NULL, 0);
	if (s == HSAKMT_STATUS_NOT_SUPPORTED) {
		PASS();
	} else {
		FAIL("expected NOT_SUPPORTED");
	}

	TEST("hsaKmtDbgRegister (stub)");
	s = hsaKmtDbgRegister(0);
	if (s == HSAKMT_STATUS_NOT_SUPPORTED) {
		PASS();
	} else {
		FAIL("expected NOT_SUPPORTED");
	}

	hsaKmtCloseKFD();
}

/* ======================================================================
 * Version query
 * ====================================================================== */
static void test_version(void)
{
	HsaVersionInfo ver;

	hsaKmtOpenKFD();

	TEST("hsaKmtGetVersion");
	memset(&ver, 0, sizeof(ver));
	HSAKMT_STATUS s = hsaKmtGetVersion(&ver);
	if (s == HSAKMT_STATUS_SUCCESS &&
	    ver.KernelInterfaceMajorVersion > 0) {
		PASS_FMT("KFD %u.%u",
			 ver.KernelInterfaceMajorVersion,
			 ver.KernelInterfaceMinorVersion);
	} else {
		FAIL("GetVersion failed or invalid version");
	}

	hsaKmtCloseKFD();
}

int main(void)
{
	printf("=== libhsakmt amdgpu_lite backend test ===\n");
	printf("=== Ported from kfdtest: KFDOpenCloseKFDTest, KFDTopologyTest ===\n\n");

	setenv("HSAKMT_DEBUG_LEVEL", "6", 0); /* INFO level */

	printf("[KFDCloseKFDTest]\n");
	test_close_a_closed_kfd();

	printf("\n[KFDOpenCloseKFDTest]\n");
	test_open_close_kfd();
	test_open_already_opened();
	test_invalid_kfd_handle();

	printf("\n[Version]\n");
	test_version();

	printf("\n[KFDTopologyTest::BasicTest]\n");
	test_topology_basic();

	printf("\n[KFDTopologyTest::GetNodePropertiesInvalidParams]\n");
	test_get_node_properties_null();

	printf("\n[KFDTopologyTest::GetNodePropertiesInvalidNodeNum]\n");
	test_get_node_properties_invalid_node();

	printf("\n[KFDTopologyTest::GetNodeMemoryProperties]\n");
	test_get_node_memory_properties();

	printf("\n[KFDTopologyTest::GpuvmApertureValidate]\n");
	test_gpuvm_aperture_validate();

	printf("\n[KFDTopologyTest::GetNodeCacheProperties]\n");
	test_get_node_cache_properties();

	printf("\n[KFDTopologyTest::GetNodeIoLinkProperties]\n");
	test_get_node_iolink_properties();

	printf("\n[Stubs]\n");
	test_stubs();

	printf("\n=== Results: %d passed, %d failed ===\n",
	       pass_count, fail_count);

	return fail_count > 0 ? 1 : 0;
}
