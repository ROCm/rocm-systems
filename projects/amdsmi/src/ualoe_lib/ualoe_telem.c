// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ualoe_telem.h"

#include <cbl_cfg/uapi.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ifoe_telemetry.h"
#include "telemetry_dataset.h"
#include "ualoe_log.h"

_Static_assert((int)CFG_TELEMETRY_CATEGORY_MAX >= (int)TELEMETRY_SRC_CATEGORY_COUNT,
               "CFG_TELEMETRY_CATEGORY_MAX must be >= TELEMETRY_SRC_CATEGORY_COUNT");

#define EXTRACT_FIELD(value, field) (((value) >> field##_LBN) & ((1ULL << field##_WIDTH) - 1))

/*
 * Overflow-safe bounds check: returns true if [_start, _start + _len)
 * exceeds _buf_size.  All arguments must be integer quantities (offsets
 * or sizes), never pointers.
 */
#ifdef __GNUC__
#define CHECK_BOUNDS(_start, _len, _buf_size)                                              \
  ({                                                                                       \
    size_t _end;                                                                           \
    __builtin_add_overflow((size_t)(_start), (size_t)(_len), &_end) || _end > (_buf_size); \
  })
#else
#define CHECK_BOUNDS(_start, _len, _buf_size) \
  ((size_t)(_len) > (_buf_size) || (size_t)(_start) > (_buf_size) - (size_t)(_len))
#endif

#define TELEM_TAG_ID(id) ((uint64_t)(id) << IFOE_TELEM_TAG_TAG_ID_LBN)

struct telem_regions {
  const char* const_base;
  const char* const_end;
  const char* dyn_base;
  const char* dyn_end;
  long page_size;
};

long ualoe_get_page_size(void) {
  static long page_size;

  if (!page_size) {
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
  }

  return page_size;
}

static void set_instance_ids(ualoe_telemetry_instance_t* instance, uint64_t tlv_tag,
                             bool netport_based, uint32_t netports_per_station) {
  uint32_t station_idx;

  station_idx = EXTRACT_FIELD(tlv_tag, IFOE_TELEM_TAG_STATION_IDX);

  if (!netport_based) {
    instance->logical_idx = station_idx;
    snprintf(instance->name.text, sizeof(instance->name.text), CFG_STATION_LABEL_FMT, station_idx);
  } else {
    uint32_t netport_rel_idx;
    uint32_t netport_logical_idx;

    netport_rel_idx = EXTRACT_FIELD(tlv_tag, IFOE_TELEM_TAG_NETPORT_REL_IDX);
    netport_logical_idx =
        CFG_NETPORT_LOGICAL_IDX(station_idx, netports_per_station, netport_rel_idx);

    instance->logical_idx = netport_logical_idx;
    snprintf(instance->name.text, sizeof(instance->name.text), CFG_NETPORT_LABEL_FMT,
             netport_logical_idx);
  }
}

static const ifoe_telem_tlv_data_t* tlv_next(const ifoe_telem_tlv_data_t* data, const char* end) {
  size_t remaining, advance;

  remaining = (size_t)(end - (const char*)data);
  if (ADD_OVERFLOW(sizeof(ifoe_telem_tlv_t), (size_t)data->tlv.len, &advance)) return NULL;
  if (advance > remaining || remaining - advance < sizeof(ifoe_telem_tlv_t)) return NULL;

  return (const ifoe_telem_tlv_data_t*)((const char*)data + advance);
}

static int ualoe_telem_parse_snapshot_hdr(const void* mmap_buf, size_t buf_size,
                                          struct telem_regions* rgn) {
  const struct cfg_telem_snapshot_hdr* snap_hdr;

  if (buf_size < sizeof(struct cfg_telem_snapshot_hdr)) {
    ualoe_log_error("telemetry_parse: buffer too small for snapshot header\n");
    return EIO;
  }

  snap_hdr = (const struct cfg_telem_snapshot_hdr*)mmap_buf;

  if (CHECK_BOUNDS(snap_hdr->const_offset, snap_hdr->const_size, buf_size) ||
      CHECK_BOUNDS(snap_hdr->dyn_offset, snap_hdr->dyn_size, buf_size)) {
    ualoe_log_error("telemetry_parse: snapshot header offsets exceed buffer\n");
    return EIO;
  }

  rgn->const_base = (const char*)mmap_buf + snap_hdr->const_offset;
  rgn->const_end = rgn->const_base + snap_hdr->const_size;
  rgn->dyn_base = (const char*)mmap_buf + snap_hdr->dyn_offset;
  rgn->dyn_end = rgn->dyn_base + snap_hdr->dyn_size;

  return 0;
}

static int ualoe_telem_validate_hdr(const struct telem_regions* rgn,
                                    const ifoe_telem_tlv_hdr_t** hdr) {
  *hdr = (const ifoe_telem_tlv_hdr_t*)(rgn->const_base + rgn->page_size);

  if ((const char*)(*hdr) + sizeof(**hdr) > rgn->const_end) {
    ualoe_log_error("telemetry_parse: header TLV outside const region\n");
    return EIO;
  }

  if ((*hdr)->magic != IFOE_TELEM_TLV_HDR_MAGIC_VALUE ||
      (*hdr)->tlv.tag != TELEM_TAG_ID(IFOE_TELEM_TAG_HEADER)) {
    ualoe_log_error("telemetry_parse: invalid header TLV magic or tag\n");
    return EIO;
  }

  return 0;
}

static int ualoe_telem_validate_locators(const ifoe_telem_tlv_hdr_t* hdr, const char* const_end,
                                         const ifoe_telem_tlv_desc_locator_t** desc_locator,
                                         const ifoe_telem_tlv_dataset_locator_t** dataset_locator) {
  size_t remaining, advance;

  *desc_locator =
      (const ifoe_telem_tlv_desc_locator_t*)((const char*)hdr + sizeof(ifoe_telem_tlv_hdr_t));

  if ((const char*)(*desc_locator) + sizeof(**desc_locator) > const_end) {
    ualoe_log_error("telemetry_parse: desc_locator outside const region\n");
    return EIO;
  }

  if ((*desc_locator)->tlv.tag != TELEM_TAG_ID(IFOE_TELEM_TAG_DESC_LOCATOR)) {
    ualoe_log_error("telemetry_parse: invalid desc_locator tag\n");
    return EIO;
  }

  remaining = (size_t)(const_end - (const char*)(*desc_locator));
  if (ADD_OVERFLOW(sizeof(ifoe_telem_tlv_t), (size_t)(*desc_locator)->tlv.len, &advance) ||
      advance > remaining || remaining - advance < sizeof(ifoe_telem_tlv_t)) {
    ualoe_log_error("telemetry_parse: dataset_locator outside const region\n");
    return EIO;
  }
  *dataset_locator =
      (const ifoe_telem_tlv_dataset_locator_t*)((const char*)(*desc_locator) + advance);

  if ((*dataset_locator)->tlv.tag != TELEM_TAG_ID(IFOE_TELEM_TAG_DATASET_LOCATOR)) {
    ualoe_log_error("telemetry_parse: invalid dataset_locator tag\n");
    return EIO;
  }

  return 0;
}

static int count_data_tlvs(const struct telem_regions* rgn,
                           const ifoe_telem_tlv_dataset_hdr_t* data_hdr, unsigned cat,
                           bool netport_based, unsigned* instance_count,
                           uint32_t* netports_per_station) {
  uint32_t max_netport_rel_idx = 0;
  const ifoe_telem_tlv_data_t* data;
  unsigned inst_idx = 0;

  data =
      (const ifoe_telem_tlv_data_t*)((const char*)data_hdr + sizeof(ifoe_telem_tlv_dataset_hdr_t));
  while (inst_idx < CFG_TELEMETRY_MAX_INSTANCES &&
         (const char*)data + sizeof(ifoe_telem_tlv_t) <= rgn->dyn_end) {
    uint8_t tag_id = EXTRACT_FIELD(data->tlv.tag, IFOE_TELEM_TAG_TAG_ID);

    if (tag_id == IFOE_TELEM_TAG_NULL) {
      data = tlv_next(data, rgn->dyn_end);
      if (!data) {
        ualoe_log_error(
            "telemetry_parse: category %u: "
            "TLV walk exceeded dyn region in count pass\n",
            cat);
        return EIO;
      }
      continue;
    }
    if (tag_id != IFOE_TELEM_TAG_DATA) break;
    if (netport_based) {
      uint32_t nri = EXTRACT_FIELD(data->tlv.tag, IFOE_TELEM_TAG_NETPORT_REL_IDX);
      if (nri > max_netport_rel_idx) max_netport_rel_idx = nri;
    }
    inst_idx++;
    data = tlv_next(data, rgn->dyn_end);
    if (!data) {
      ualoe_log_error(
          "telemetry_parse: category %u: "
          "TLV walk exceeded dyn region in count pass\n",
          cat);
      return EIO;
    }
  }

  *instance_count = inst_idx;
  *netports_per_station = netport_based ? (max_netport_rel_idx + 1) : 1;
  return 0;
}

static int populate_instances(ualoe_telemetry_dataset_t* dataset, const struct telem_regions* rgn,
                              const ifoe_telem_tlv_dataset_hdr_t* data_hdr,
                              const ifoe_telem_tlv_desc_t* desc, unsigned cat, bool netport_based,
                              unsigned item_count, uint32_t netports_per_station) {
  const ifoe_telem_tlv_data_t* data;
  unsigned inst_idx;

  data =
      (const ifoe_telem_tlv_data_t*)((const char*)data_hdr + sizeof(ifoe_telem_tlv_dataset_hdr_t));
  for (inst_idx = 0; inst_idx < dataset->instance_count; inst_idx++) {
    ualoe_telemetry_instance_t* instance;
    unsigned int k;

    /* Skip any NULL TLVs between DATA TLVs */
    while ((const char*)data + sizeof(ifoe_telem_tlv_t) <= rgn->dyn_end &&
           EXTRACT_FIELD(data->tlv.tag, IFOE_TELEM_TAG_TAG_ID) == IFOE_TELEM_TAG_NULL) {
      data = tlv_next(data, rgn->dyn_end);
      if (!data) {
        ualoe_log_error(
            "telemetry_parse: category %u: "
            "TLV walk exceeded dyn region in data pass\n",
            cat);
        return EIO;
      }
    }

    if ((const char*)data + sizeof(ifoe_telem_tlv_t) > rgn->dyn_end ||
        EXTRACT_FIELD(data->tlv.tag, IFOE_TELEM_TAG_TAG_ID) != IFOE_TELEM_TAG_DATA) {
      ualoe_log_error(
          "telemetry_parse: category %u: "
          "unexpected tag in data pass\n",
          cat);
      return EIO;
    }

    if (data->tlv.len < item_count * sizeof(uint64_t)) {
      ualoe_log_error(
          "telemetry_parse: category %u: "
          "DATA TLV too short for %u items\n",
          cat, item_count);
      return EIO;
    }

    if ((const char*)&data->values[item_count] > rgn->dyn_end) {
      ualoe_log_error(
          "telemetry_parse: category %u: "
          "data values exceed dyn region\n",
          cat);
      return EIO;
    }

    instance = &dataset->instances[inst_idx];
    set_instance_ids(instance, data->tlv.tag, netport_based, netports_per_station);
    instance->item_count = item_count;

    for (k = 0; k < IFOE_TELEMETRY_ID_COUNT && k < item_count; k++) {
      instance->items[k].id = desc->ids[k];
      instance->items[k].value = data->values[k];
    }

    data = tlv_next(data, rgn->dyn_end);
    if (!data && inst_idx + 1 < dataset->instance_count) {
      ualoe_log_error(
          "telemetry_parse: category %u: "
          "TLV walk exceeded dyn region in data pass\n",
          cat);
      return EIO;
    }
  }

  return 0;
}

static int parse_category(const struct telem_regions* rgn, unsigned cat,
                          const ifoe_telem_tlv_desc_locator_t* desc_locator,
                          const ifoe_telem_tlv_dataset_locator_t* dataset_locator,
                          ualoe_telemetry_dataset_t* dataset) {
  size_t const_size, dyn_size, offset;
  uint32_t netports_per_station;
  const ifoe_telem_tlv_dataset_hdr_t* data_hdr;
  const ifoe_telem_tlv_desc_t* desc;
  unsigned item_count;
  bool netport_based;
  int rc;

  /* Find desc TLV in const region */
  const_size = (size_t)(rgn->const_end - rgn->const_base);
  if (ADD_OVERFLOW((size_t)rgn->page_size, (size_t)desc_locator->descs[cat], &offset) ||
      offset > const_size || const_size - offset < sizeof(*desc)) {
    ualoe_log_error("telemetry_parse: category %u: desc outside const region\n", cat);
    return EIO;
  }
  desc = (const ifoe_telem_tlv_desc_t*)(rgn->const_base + offset);

  if (EXTRACT_FIELD(desc->tlv.tag, IFOE_TELEM_TAG_TAG_ID) != IFOE_TELEM_TAG_DESC) {
    ualoe_log_error("telemetry_parse: category %u: invalid desc tag\n", cat);
    return EIO;
  }

  dataset->category = EXTRACT_FIELD(desc->tlv.tag, IFOE_TELEM_TAG_CATEGORY);
  item_count = desc->tlv.len / sizeof(uint64_t);
  if (item_count > IFOE_TELEMETRY_ID_COUNT) item_count = IFOE_TELEMETRY_ID_COUNT;

  if ((const char*)&desc->ids[item_count] > rgn->const_end) {
    ualoe_log_error("telemetry_parse: category %u: desc ids exceed const region\n", cat);
    return EIO;
  }

  /* Find dataset header TLV in dyn region */
  dyn_size = (size_t)(rgn->dyn_end - rgn->dyn_base);
  if (ADD_OVERFLOW((size_t)rgn->page_size, (size_t)dataset_locator->datasets[cat], &offset) ||
      offset > dyn_size || dyn_size - offset < sizeof(*data_hdr)) {
    ualoe_log_error("telemetry_parse: category %u: dataset_hdr outside dyn region\n", cat);
    return EIO;
  }
  data_hdr = (const ifoe_telem_tlv_dataset_hdr_t*)(rgn->dyn_base + offset);

  if (EXTRACT_FIELD(data_hdr->tlv.tag, IFOE_TELEM_TAG_TAG_ID) != IFOE_TELEM_TAG_DATASET_HDR) {
    ualoe_log_error("telemetry_parse: category %u: invalid dataset_hdr tag\n", cat);
    return EIO;
  }

  netport_based = (cat == CFG_TELEMETRY_CATEGORY_PFC || cat == CFG_TELEMETRY_CATEGORY_NETPORT ||
                   cat == CFG_TELEMETRY_CATEGORY_DERIVED_NETPORT);

  rc = count_data_tlvs(rgn, data_hdr, cat, netport_based, &dataset->instance_count,
                       &netports_per_station);
  if (rc) return rc;

  dataset->generation_count = data_hdr->gen_count;
  dataset->timestamp.tv_sec = data_hdr->timestamp / 1000;
  dataset->timestamp.tv_nsec = (data_hdr->timestamp % 1000) * 1000000;

  rc = populate_instances(dataset, rgn, data_hdr, desc, cat, netport_based, item_count,
                          netports_per_station);
  if (rc) return rc;

  return 0;
}

int ualoe_telem_parse(const void* mmap_buf, size_t buf_size, unsigned category_mask,
                      ualoe_telemetry_t* telemetry) {
  const ifoe_telem_tlv_dataset_locator_t* dataset_locator;
  const ifoe_telem_tlv_desc_locator_t* desc_locator;
  const ifoe_telem_tlv_hdr_t* hdr;
  struct telem_regions rgn;
  unsigned cat;
  int rc;

  rgn.page_size = ualoe_get_page_size();

  rc = ualoe_telem_parse_snapshot_hdr(mmap_buf, buf_size, &rgn);
  if (rc) return rc;

  rc = ualoe_telem_validate_hdr(&rgn, &hdr);
  if (rc) return rc;

  rc = ualoe_telem_validate_locators(hdr, rgn.const_end, &desc_locator, &dataset_locator);
  if (rc) return rc;

  for (cat = 0; cat < TELEMETRY_SRC_CATEGORY_COUNT && cat < UALOE_TELEMETRY_CATEGORY_MAX; cat++) {
    if (category_mask & (1U << cat)) {
      if (telemetry->datasets[cat] == NULL) {
        ualoe_log_error("No dataset allocated for category %d\n", cat);
        return EINVAL;
      } else if (desc_locator->descs[cat] == 0) {
        ualoe_log_error(
            "Firmware has not provided data for "
            "category %d\n",
            cat);
        return EINVAL;
      } else if (dataset_locator->datasets[cat] == 0) {
        ualoe_log_error(
            "Firmware has not provided dataset for "
            "category %d\n",
            cat);
        return EINVAL;
      }
    } else {
      if (telemetry->datasets[cat]) telemetry->datasets[cat]->instance_count = 0;
      continue;
    }

    rc = parse_category(&rgn, cat, desc_locator, dataset_locator, telemetry->datasets[cat]);
    if (rc) return rc;
  }

  return 0;
}
