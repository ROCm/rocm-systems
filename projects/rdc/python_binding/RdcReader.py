import os,time
from rdc_bootstrap import *
from RdcUtil import RdcUtil
from typing import Dict

default_field_ids = [
        rdc_field_t.RDC_FI_GPU_MEMORY_USAGE,
        rdc_field_t.RDC_FI_GPU_MEMORY_TOTAL,
        rdc_field_t.RDC_FI_GPU_MM_ENC_UTIL,
        rdc_field_t.RDC_FI_GPU_MM_DEC_UTIL,
        rdc_field_t.RDC_FI_GPU_MEMORY_ACTIVITY,
        rdc_field_t.RDC_FI_GPU_MEMORY_MAX_BANDWIDTH,
        rdc_field_t.RDC_FI_GPU_MEMORY_CUR_BANDWIDTH,
        rdc_field_t.RDC_FI_OAM_ID,
        rdc_field_t.RDC_FI_POWER_USAGE,
        rdc_field_t.RDC_FI_GPU_CLOCK,
        rdc_field_t.RDC_FI_GPU_UTIL,
        rdc_field_t.RDC_FI_GPU_TEMP,
        rdc_field_t.RDC_FI_GPU_MEMORY_USAGE
]

default_unit_coverter = {
        rdc_field_t.RDC_FI_GPU_MEMORY_USAGE: 0.000001, # MegaBytes
        rdc_field_t.RDC_FI_GPU_MEMORY_TOTAL: 0.000001, # MegaBytes
        rdc_field_t.RDC_FI_POWER_USAGE: 0.000001, # Watts
        rdc_field_t.RDC_FI_GPU_CLOCK: 0.000001, # MHz
        rdc_field_t.RDC_FI_GPU_TEMP: 0.001, # degree
}

class RdcReader:
    # To run the RDC in embedded mode, set the ip_port = None
    def __init__(self, ip_port = "localhost:50051", field_ids = default_field_ids,
            unit_converter: Dict[int, float] = default_unit_coverter,
            update_freq = 10000000, max_keep_age = 3600.0 , max_keep_samples = 1000,
            field_group_name = "rdc_reader_field_group", gpu_group_name = "rdc_reader_gpu_group",
            gpu_indexes = None, root_ca = "/etc/rdc/client/certs/rdc_cacert.pem",
            client_cert = "/etc/rdc/client/certs/rdc_client_cert.pem",
            client_key = "/etc/rdc/client/private/rdc_client_cert.key"):
        result = rdc.rdc_init(0)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            raise Exception("RdcReader init fail: " + str(result))

        self.rdc_util = RdcUtil()

        self.unit_converter = unit_converter
        self.rdc_handle = c_void_p()

        self.is_standalone = True
        if not ip_port:  # embedded
            self.is_standalone = False
            result = rdc.rdc_start_embedded(rdc_operation_mode_t.RDC_OPERATION_MODE_AUTO, self.rdc_handle)
            if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                raise Exception("RdcReader start as embedded fail: " + str(result))
        else: # standalone
            if root_ca == None or client_cert == None or client_key == None:
                with_auth = False
                root_ca_str = client_cert_str = client_key_str = None
            else:
                with_auth = True
                root_ca_str = self.rdc_util.read_file(root_ca)
                client_cert_str = self.rdc_util.read_file(client_cert)
                client_key_str = self.rdc_util.read_file(client_key)

            result = rdc.rdc_connect(ip_port.encode('utf-8'), self.rdc_handle, root_ca_str, client_cert_str, client_key_str)
            if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                raise Exception("RdcReader standalone auth(" + str(with_auth) + ") connect to " + ip_port+ " fail: " + str(result))

        # Create the GPU group
        self.gpu_group_name = gpu_group_name.encode()
        if gpu_indexes == None:
            self.gpu_indexes = self.rdc_util.get_all_gpu_indexes(self.rdc_handle)
        else:
            self.gpu_indexes = []
            for idx in gpu_indexes:
                idx_str = str(idx)
                encoded = idx_str.encode("utf-8")
                phys_gpu = ctypes.c_uint32()
                part_idx = ctypes.c_uint32()
                if rdc.rdc_is_partition_string(encoded):
                    rc = rdc.rdc_parse_partition_string(encoded, ctypes.byref(phys_gpu), ctypes.byref(part_idx))
                    if not rc:
                        raise Exception("Rdc failed to parse partition string")
                    info = rdc_entity_info_t()
                    info.device_type   = 0 #RDC_DEVICE_TYPE_GPU
                    info.entity_role   = 1 #RDC_DEVICE_ROLE_PARTITION
                    info.instance_index = part_idx
                    info.device_index   = phys_gpu
                    entity = rdc.rdc_get_entity_index_from_info(info)
                    self.gpu_indexes.append(entity)
                else:
                    self.gpu_indexes.append(int(idx_str))
        self.gpu_group_id, gpu_group_created = self.rdc_util.create_gpu_group(self.rdc_handle, self.gpu_group_name, self.gpu_indexes)

        # Create the field group
        self.field_ids = field_ids
        self.field_group_name = field_group_name.encode()
        self.field_group_id, field_group_created = self.rdc_util.create_field_group(self.rdc_handle, self.field_group_name, self.field_ids)

        # Watch the fields
        self.update_freq = update_freq
        self.max_keep_age = max_keep_age
        self.max_keep_samples = max_keep_samples

        # Unwatch first to clean up what left from last run
        rdc.rdc_field_unwatch(self.rdc_handle, self.gpu_group_id, self.field_group_id)
        result = rdc.rdc_field_watch(self.rdc_handle, self.gpu_group_id,
                 self.field_group_id, self.update_freq, self.max_keep_age, self.max_keep_samples);

        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
               raise Exception("RdcReader fail to watch group " + str(self.gpu_group_id) + ", field group " + str(self.field_group_id) + ":" + str(result))

    # Process the fields periodically
    def process(self):
        has_succeed = False
        for gindex in self.gpu_indexes:
            for fid in self.field_ids:
                value = rdc_field_value()
                result = rdc.rdc_field_get_latest_value(self.rdc_handle,
                        gindex, fid, value)

                if rdc_status_t(result) == rdc_status_t.RDC_ST_OK:
                    # Convert the unit
                    if self.unit_converter != None and fid in self.unit_converter:
                        if value.type.value == rdc_field_type_t.INTEGER:
                            value.value.l_int = int(value.value.l_int * self.unit_converter[fid])
                        if value.type.value == rdc_field_type_t.DOUBLE:
                            value.value.dbl = int(value.value.dbl * self.unit_converter[fid])
                    # convert from double to l_int
                    if value.type.value == rdc_field_type_t.DOUBLE:
                        value.value.l_int = int(value.value.dbl)
                    self.handle_field(gindex, value)
                    has_succeed = True

        self.process_other_fields()

        if len(self.gpu_indexes) != 0 and len(self.field_ids) != 0 and has_succeed == False:
            self.try_reconnect()

    def process_other_fields(self):
        pass

    def try_reconnect(self):
        if self.is_standalone == False:
            return

        try:
            # When rdcd restart, the GPU and field group need to be re-created.
            self.gpu_group_id, gpu_group_created = self.rdc_util.create_gpu_group(self.rdc_handle, self.gpu_group_name, self.gpu_indexes)
            self.field_group_id, field_group_created = self.rdc_util.create_field_group(self.rdc_handle, self.field_group_name, self.field_ids)

            # rdcd restart requires to watch the group again
            if gpu_group_created or field_group_created:
                result = rdc.rdc_field_watch(self.rdc_handle, self.gpu_group_id,
                        self.field_group_id, self.update_freq, self.max_keep_age, self.max_keep_samples);
                if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                     raise Exception("RdcReader fail to watch group " + str(self.gpu_group_id) + ", field group " + str(self.field_group_id) + ":" + str(result))
        except Exception as e:
            print(e)


    def handle_field(self, gpu_index, value):

        info = rdc.rdc_get_info_from_entity_index(gpu_index)

        if info.entity_role == 1: #RDC_DEVICE_ROLE_PARTITION_INSTANCE
            gpu_str = f"g{info.device_index}.{info.instance_index}"
        else:
            gpu_str = str(info.device_index)

        field_name = self.rdc_util.field_id_string(value.field_id)
        print("%d %s:%d %s:%d" % (value.ts, gpu_str, value.field_id.value, field_name, value.value.l_int))


if __name__ == '__main__':
    # Run the reader in embedded mode
    reader = RdcReader(ip_port=None, update_freq=1000000)
    while True:
        time.sleep(1)
        reader.process()
