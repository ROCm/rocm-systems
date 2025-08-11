from rdc_bootstrap import *

class RdcUtil:
    def __init__(self):
        pass

    def get_all_gpu_indexes(self, rdc_handle):
        gpu_count = c_uint32()
        gpu_index_list = (c_uint32 * RDC_MAX_NUM_DEVICES)()

        result = rdc.rdc_device_get_all(rdc_handle, gpu_index_list, gpu_count)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            raise Exception("Fail to get all GPus")
        gpu_indexes = []
        for index in range(gpu_count.value):
            gpu_indexes.append(gpu_index_list[index])
        return gpu_indexes

    def get_all_gpu_groups(self, rdc_handle):
        all_groups = {}
        group_count = c_uint32()
        gpu_group_list = (c_uint32 * RDC_MAX_NUM_GROUPS)()

        result = rdc.rdc_group_get_all_ids(rdc_handle, gpu_group_list, group_count)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            raise Exception("Fail to get all groups")
        for index in range(group_count.value):
            group_id = gpu_group_list[index]
            group_info = rdc_group_info_t()
            result = rdc.rdc_group_gpu_get_info(rdc_handle, group_id, group_info)
            all_groups[group_id] = group_info

        return all_groups


    # Create gpu group if not exists
    # Return <gpu_group_id, is_created>
    def create_gpu_group(self, rdc_handle, gpu_group_name, gpu_indexes):
        # Can we reuse the exists one?
        all_groups = self.get_all_gpu_groups(rdc_handle)
        for id,group_info in all_groups.items():
            group_name = group_info.group_name.decode('utf-8')
            list_gpu_indexes = list(group_info.entity_ids[:group_info.count])
            if group_name == gpu_group_name:
                # Reuse existing group
                if list_gpu_indexes == gpu_indexes:
                    return id, False
                else: # delete old group
                    result = rdc.rdc_group_gpu_destroy(rdc_handle, id)
                    if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                        raise Exception("Fail to delete the GPU group")

        #Create new gpu group
        gpu_group_id = c_uint32()
        result = rdc.rdc_group_gpu_create(rdc_handle, rdc_group_type_t.RDC_GROUP_EMPTY, gpu_group_name, gpu_group_id)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            raise Exception("Fail to create the GPU group " + group_name)

        #Add GPU index to the group
        for gpu in gpu_indexes:
            result = rdc.rdc_group_gpu_add(rdc_handle, gpu_group_id, gpu)
            if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                raise Exception("Fail to add GPU index " + str(gpu) + " to group " + str(gpu_group_id))

        return gpu_group_id, True

    def create_field_group(self, rdc_handle, field_group_name, field_ids):
        # Do we need to recreate the field group?
        field_group_id_list = (rdc_field_grp_t * RDC_MAX_FIELD_IDS_PER_FIELD_GROUP)()
        field_group_count = c_uint32()
        result = rdc.rdc_group_field_get_all_ids(rdc_handle, field_group_id_list, field_group_count)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            raise Exception("Fail to get all field group")
        for index in range(field_group_count.value):
            group_info = rdc_field_group_info_t()
            result = rdc.rdc_group_field_get_info(rdc_handle, field_group_id_list[index], pointer(group_info))
            if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                raise Exception("Fail to get field group " + str(field_group_id_list[index]) + " info")
            if group_info.group_name.decode("utf-8") == field_group_name:
                field_ids_ori = [ e.value for  e in group_info.field_ids[:group_info.count] ]
                # reuse the old field group
                if (field_ids == field_ids_ori):
                    return field_group_id_list[index], False
                else:
                    result = rdc.rdc_group_field_destroy(rdc_handle, field_group_id_list[index])
                    if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                        raise Exception("Fail to delete field group " + str(field_group_id_list[index]))

        #Create new field group
        fields_c_ids = []
        for f in field_ids:
            fields_c_ids.append(rdc_field_t(f))
        c_ids = ( rdc_field_t * len(field_ids))(*fields_c_ids)

        field_group_id = c_uint32()
        result = rdc.rdc_group_field_create(rdc_handle, len(field_ids), c_ids, field_group_name, field_group_id)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            raise Exception("Fail to create field group " + field_group_name.decode("utf-8") +": " + str(result))

        return field_group_id, True

    def field_id_string(self, field_id):
        return rdc.field_id_string(field_id).decode("utf-8")

    def read_file(self, file_name):
        try:
            with open(file_name, 'r') as file:
              return file.read().encode('utf-8')
        except Exception as e:
            print("Fail to read " + file_name + ":" + str(e))

        return None
