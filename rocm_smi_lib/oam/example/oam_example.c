#include <stdio.h>
#include "oam/oam_mapi.h"
#include "oam/amd_oam.h"

const oam_ops_t amd_oam_ops = {
	.init = amdoam_init,
	.free = amdoam_free,
//	.get_mapi_version = amdoam_get_mapi_version,
	.discover_devices = amdoam_discover_devices,
	.get_dev_properties = amdoam_get_dev_properties,
	.get_pci_properties = amdoam_get_pci_properties,
	.get_sensors_count = amdoam_get_sensors_count,
	.get_error_description = amdoam_get_error_description,
	.get_sensors_info = amdoam_get_sensors_info,
};

static int get_sensor_info(uint32_t device_id, oam_sensor_type_t type,
			uint32_t num_sensors, char unit[]) {
	uint32_t j;
	oam_sensor_info_t *sensor_info = calloc(num_sensors,
	 sizeof(oam_sensor_info_t));
	if (!sensor_info) {
		printf("Allocating power_info failed\n");
		return -1;
	}
	amd_oam_ops.get_sensors_info(device_id, type, num_sensors, sensor_info);
	for ( j = 0; j < num_sensors ; j++) {
		printf("\tSensor Name : %s \n", sensor_info[j].sensor_name);
		printf("\tSensor Type : %d \n", sensor_info[j].sensor_type);
		printf("\tSensor Value : %ld %s\n", sensor_info[j].value, unit);
  	}
	free(sensor_info);
	printf("\t**************************************\n");
	return 0;
}

int main()
{
	uint32_t dev_cnt = 0;
	oam_mapi_version_t version;
	oam_dev_properties_t *devs_prop;
	int i;
	oam_pci_info_t pci_info;
	oam_sensor_count_t sensor_count;
	const char *string;

	if (amd_oam_ops.init()) {
		printf("init failed\n");
		return -1;
	}

//	amd_oam_ops.get_mapi_version(&version);
	if (!amd_oam_ops.discover_devices(&dev_cnt))
		printf("%u AMD devices are discovered\n", dev_cnt);
	if (!dev_cnt) {
		printf("No devices are  found.\n");
		return amd_oam_ops.free();
	}
	devs_prop = calloc(dev_cnt, sizeof(oam_dev_properties_t));
	if (!devs_prop) {
		printf("Allocating dev_prop failed\n");
		return amd_oam_ops.free();
	}

	amd_oam_ops.get_dev_properties(dev_cnt, devs_prop);
	for (i = 0; i < dev_cnt; i++) {
		printf("Device %d:\n", i);
		printf("\tdevice id %d\n", devs_prop[i].device_id);
		printf("\tdevice_vendor %s\n", devs_prop[i].device_vendor);
		printf("\tdevice_name %s\n", devs_prop[i].device_name);
		printf("\tsku_name %s\n", devs_prop[i].sku_name);
		printf("\tboard_name %s\n", devs_prop[i].board_name);
		printf("\tboard_revision %s\n", devs_prop[i].board_revision);
		printf("\tboard_serial_number %s\n",
					 devs_prop[i].board_serial_number);

		if (!amd_oam_ops.get_pci_properties(
			devs_prop[i].device_id, &pci_info)) {
			printf("\tPCI domain : 0x%d \n", pci_info.domain);
			printf("\tPCI bus : 0x%d \n", pci_info.bus);
			printf("\tPCI device : 0x%d \n", pci_info.device);
			printf("\tPCI function : 0x%d \n", pci_info.function);
		}

		printf("\t**************************************\n");
		if (amd_oam_ops.get_sensors_count(
			devs_prop[i].device_id, &sensor_count))
			continue;
		printf("\tNumber of Power Sensors : %d \n",
				    sensor_count.num_power_sensors);
		if (get_sensor_info(devs_prop[i].device_id,OAM_SENSOR_TYPE_POWER,
		 sensor_count.num_power_sensors, "uW"))
		 	goto failure;

		printf("\tNumber of Voltage Sensors : %d \n",
		            sensor_count.num_voltage_sensors);
		if (get_sensor_info(devs_prop[i].device_id, OAM_SENSOR_TYPE_VOLTAGE,
		 sensor_count.num_voltage_sensors, "mV"))
		 	goto failure;

		printf("\tNumber of Current Sensors : %d \n",
		            sensor_count.num_current_sensors);
		if (get_sensor_info(devs_prop[i].device_id, OAM_SENSOR_TYPE_CURRENT,
		 sensor_count.num_current_sensors, "A"))
			goto failure;

		printf("\tNumber of Temperature Sensors : %d \n",
		            sensor_count.num_temperature_sensors);
		if (get_sensor_info(devs_prop[i].device_id, OAM_SENSOR_TYPE_TEMP,
		 sensor_count.num_temperature_sensors, "mC"))
		 	goto failure;

		printf("\tNumber of Fan Sensors : %d \n", sensor_count.num_fans);
		if (get_sensor_info(devs_prop[i].device_id, OAM_SENSOR_TYPE_FAN_SPEED,
		 sensor_count.num_fans, "rpm"))
			goto failure;
	}

	amd_oam_ops.get_error_description(1, &string);
	printf("error code 1: %s\n", string);

failure:
	free(devs_prop);
	amd_oam_ops.free();

	return 0;
}

