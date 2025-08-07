from RdcReader import RdcReader
from rdc_bootstrap import *
import collectd

default_field_ids = [
        rdc_field_t.RDC_FI_GPU_MEMORY_USAGE,
        rdc_field_t.RDC_FI_GPU_MEMORY_TOTAL,
        rdc_field_t.RDC_FI_POWER_USAGE,
        rdc_field_t.RDC_FI_GPU_CLOCK,
        rdc_field_t.RDC_FI_GPU_UTIL,
        rdc_field_t.RDC_FI_GPU_TEMP,
        rdc_field_t.RDC_FI_GPU_MEMORY_USAGE,
]


class CollectdReader(RdcReader):
    def __init__(self, rdc_ip_port, field_ids, update_freq, max_keep_age, max_keep_samples,
            gpu_indexes, rdc_unauth):
        group_name = "rdc_collectd_plugin_group"
        field_group_name = "rdc_collectd_plugin_fieldgroup"
        if rdc_unauth:
            RdcReader.__init__(self, ip_port = rdc_ip_port, field_ids = field_ids, update_freq=update_freq,
                    max_keep_age = max_keep_age, max_keep_samples = max_keep_samples,
                    gpu_indexes = gpu_indexes, field_group_name = field_group_name, gpu_group_name = group_name, root_ca = None)
        else:
             RdcReader.__init__(self, ip_port = rdc_ip_port, field_ids = field_ids, update_freq=update_freq,
                    max_keep_age = max_keep_age, max_keep_samples = max_keep_samples,
                    gpu_indexes = gpu_indexes, field_group_name = field_group_name, gpu_group_name = group_name)

    def handle_field(self, gpu_index, value):
        PLUGIN_NAME = "rdc_collectd"
        field_name = self.rdc_util.field_id_string(value.field_id).lower()
        collectd.Values(plugin=PLUGIN_NAME,
                    type_instance= field_name,
                    type="gauge",
                    values=[value.value.l_int]).dispatch()

g_reader = None

def config_func(config):
    global g_reader

    embedded = False # enable embedded if no rdcd
    rdc_ip_port = "localhost:50051" # rdcd listen address
    field_ids = default_field_ids # The fields to watch
    update_freq = 10  # 10 seconds
    max_keep_age = 3600    # 1 hour
    max_keep_samples = 1000 # The max samples to keep for each field
    gpu_indexes = None # All GPus
    unauth = False  # Enable auth by default

    # Parse configure parameters
    for node in config.children:
        key = node.key.lower()
        if len(node.values) <= 0:
            print("Missing value in configure " + key)
            continue

        val = node.values[0]
        if key == 'embedded' and val == True:
            embedded = True
        if key == 'rdc_ip_port':
            rdc_ip_port = val
        if key == 'unauth':
            unauth = val
        if key == 'field_ids':
            field_ids = []
            for f in node.values:
                field_id = rdc.get_field_id_from_name(str.encode(f))
                if field_id.value == rdc_field_t.RDC_FI_INVALID:
                    print("Invalid field '%s' will be ignored." % (f))
                else:
                    field_ids.append(field_id.value)
        if key == 'update_freq':
            update_freq = int(val)
        if key == 'max_keep_age':
            max_keep_age = int(max_keep_age)
        if key == 'max_keep_samples':
            max_keep_samples = int(max_keep_samples)
        if key == 'gpu_indexes':
            gpu_indexes = [int(x) for x in node.values]

    if embedded:
        rdc_ip_port = None
    g_reader = CollectdReader(rdc_ip_port, field_ids, update_freq*1000000,
            max_keep_age, max_keep_samples, gpu_indexes, unauth)

def read_callback(data=None):
    global g_reader
    g_reader.process()

collectd.register_config(config_func)
collectd.register_read(read_callback)
