// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fwupd_carveout.h"

#include <dlfcn.h>
#include <strings.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fwupd_carveout_internal.h"

// The UMA "carveout" BIOS setting is owned by the fwupd system daemon. amd-smi
// reads and writes it over the daemon's stable D-Bus interface, loading the
// reference D-Bus client (libdbus-1) lazily with dlopen() at runtime. As a
// result the AMD SMI library carries no build-, link-, or package-time
// dependency for this feature: if libdbus-1 or the fwupd daemon is not present,
// the calls report AMDSMI_STATUS_NOT_SUPPORTED and the amdgpu sysfs node stays
// the sole interface.
//
//   org.freedesktop.fwupd  /  org.freedesktop.fwupd
//     GetBiosSettings() -> aa{sv}   (options + current value)
//     SetBiosSettings(a{ss})        (write; PolicyKit-brokered)

using amd::smi::detail::BiosSetting;

namespace {

// --- Minimal libdbus-1 ABI ---------------------------------------------------
// Only the small, long-stable slice of libdbus-1 that we call is declared here,
// so no D-Bus development headers are needed to build. Connections and messages
// are opaque handles. The error and iterator are stack structures owned by
// libdbus; we treat them as opaque, generously sized buffers and never inspect
// their internals -- except the two leading pointer fields of the error (its
// public name/message), whose position is fixed by the ABI.
extern "C" {
struct DBusConnection;
struct DBusMessage;
using dbus_bool_t = unsigned int;

struct DbusError {
  const char* name;
  const char* message;
  void* reserved[6];  // >= sizeof(real DBusError); only name/message are read
};
struct DbusIter {
  void* reserved[16];  // >= sizeof(real DBusMessageIter); never inspected
};
}  // extern "C"

// D-Bus argument type codes: ASCII values fixed by the wire protocol.
constexpr int kBusSystem = 1;
constexpr int kTypeString = 's';
constexpr int kTypeArray = 'a';
constexpr int kTypeBoolean = 'b';
constexpr int kTypeDictEntry = 'e';

constexpr const char* kService = "org.freedesktop.fwupd";
constexpr const char* kPath = "/";
constexpr const char* kInterface = "org.freedesktop.fwupd";
constexpr const char* kKeyId = "BiosSettingId";
constexpr const char* kKeyName = "Name";
constexpr const char* kKeyCurrent = "BiosSettingCurrentValue";
constexpr const char* kKeyReadOnly = "BiosSettingReadOnly";
constexpr const char* kKeyValues = "BiosSettingPossibleValues";

// Resolved libdbus-1 entry points.
struct Dbus {
  void (*error_init)(DbusError*);
  dbus_bool_t (*error_is_set)(const DbusError*);
  void (*error_free)(DbusError*);
  DBusConnection* (*bus_get_private)(int, DbusError*);
  void (*conn_set_exit_on_disconnect)(DBusConnection*, dbus_bool_t);
  void (*conn_close)(DBusConnection*);
  void (*conn_unref)(DBusConnection*);
  DBusMessage* (*send_block)(DBusConnection*, DBusMessage*, int, DbusError*);
  DBusMessage* (*msg_new_call)(const char*, const char*, const char*, const char*);
  void (*msg_unref)(DBusMessage*);
  dbus_bool_t (*iter_init)(DBusMessage*, DbusIter*);
  void (*iter_init_append)(DBusMessage*, DbusIter*);
  void (*iter_recurse)(DbusIter*, DbusIter*);
  int (*iter_get_arg_type)(DbusIter*);
  void (*iter_get_basic)(DbusIter*, void*);
  dbus_bool_t (*iter_next)(DbusIter*);
  dbus_bool_t (*iter_open)(DbusIter*, int, const char*, DbusIter*);
  dbus_bool_t (*iter_close)(DbusIter*, DbusIter*);
  dbus_bool_t (*iter_append)(DbusIter*, int, const void*);
};

// Lazily load libdbus-1 and resolve the needed symbols exactly once.
const Dbus* LoadDbus() {
  static Dbus dbus;
  static const bool ok = [] {
    void* h = dlopen("libdbus-1.so.3", RTLD_LAZY | RTLD_LOCAL);
    if (h == nullptr) return false;
    auto load = [&](auto& fp, const char* name) {
      void* p = dlsym(h, name);
      if (p == nullptr) return false;
      std::memcpy(&fp, &p, sizeof(p));
      return true;
    };
    const bool good =
        load(dbus.error_init, "dbus_error_init") && load(dbus.error_is_set, "dbus_error_is_set") &&
        load(dbus.error_free, "dbus_error_free") &&
        load(dbus.bus_get_private, "dbus_bus_get_private") &&
        load(dbus.conn_set_exit_on_disconnect, "dbus_connection_set_exit_on_disconnect") &&
        load(dbus.conn_close, "dbus_connection_close") &&
        load(dbus.conn_unref, "dbus_connection_unref") &&
        load(dbus.send_block, "dbus_connection_send_with_reply_and_block") &&
        load(dbus.msg_new_call, "dbus_message_new_method_call") &&
        load(dbus.msg_unref, "dbus_message_unref") &&
        load(dbus.iter_init, "dbus_message_iter_init") &&
        load(dbus.iter_init_append, "dbus_message_iter_init_append") &&
        load(dbus.iter_recurse, "dbus_message_iter_recurse") &&
        load(dbus.iter_get_arg_type, "dbus_message_iter_get_arg_type") &&
        load(dbus.iter_get_basic, "dbus_message_iter_get_basic") &&
        load(dbus.iter_next, "dbus_message_iter_next") &&
        load(dbus.iter_open, "dbus_message_iter_open_container") &&
        load(dbus.iter_close, "dbus_message_iter_close_container") &&
        load(dbus.iter_append, "dbus_message_iter_append_basic");
    if (!good) {
      dlclose(h);
      return false;
    }
    return true;
  }();
  return ok ? &dbus : nullptr;
}

void ReadVariant(const Dbus& d, DbusIter* kv, BiosSetting* s, const char* key) {
  DbusIter var;
  d.iter_recurse(kv, &var);
  const int t = d.iter_get_arg_type(&var);
  if (t == kTypeString) {
    const char* v = nullptr;
    d.iter_get_basic(&var, &v);
    const std::string val = v ? v : "";
    if (std::strcmp(key, kKeyId) == 0) {
      s->id = val;
    } else if (std::strcmp(key, kKeyName) == 0) {
      s->name = val;
    } else if (std::strcmp(key, kKeyCurrent) == 0) {
      s->current = val;
    }
  } else if (t == kTypeBoolean) {
    dbus_bool_t b = 0;
    d.iter_get_basic(&var, &b);
    if (std::strcmp(key, kKeyReadOnly) == 0) s->read_only = (b != 0);
  } else if (t == kTypeArray && std::strcmp(key, kKeyValues) == 0) {
    DbusIter arr;
    d.iter_recurse(&var, &arr);
    while (d.iter_get_arg_type(&arr) == kTypeString) {
      const char* v = nullptr;
      d.iter_get_basic(&arr, &v);
      s->values.emplace_back(v ? v : "");
      d.iter_next(&arr);
    }
  }
}

bool GetSettings(const Dbus& d, DBusConnection* conn, std::vector<BiosSetting>* out) {
  DbusError err;
  d.error_init(&err);
  DBusMessage* msg = d.msg_new_call(kService, kPath, kInterface, "GetBiosSettings");
  if (msg == nullptr) return false;
  DBusMessage* reply = d.send_block(conn, msg, 5000, &err);
  d.msg_unref(msg);
  if (d.error_is_set(&err)) {
    d.error_free(&err);
    return false;
  }
  if (reply == nullptr) return false;
  DbusIter it;
  if (d.iter_init(reply, &it) && d.iter_get_arg_type(&it) == kTypeArray) {
    DbusIter sit;
    d.iter_recurse(&it, &sit);
    while (d.iter_get_arg_type(&sit) == kTypeArray) {
      BiosSetting s;
      DbusIter dit;
      d.iter_recurse(&sit, &dit);
      while (d.iter_get_arg_type(&dit) == kTypeDictEntry) {
        DbusIter kv;
        d.iter_recurse(&dit, &kv);
        const char* key = nullptr;
        d.iter_get_basic(&kv, &key);
        d.iter_next(&kv);
        ReadVariant(d, &kv, &s, key ? key : "");
        d.iter_next(&dit);
      }
      out->push_back(std::move(s));
      d.iter_next(&sit);
    }
  }
  d.msg_unref(reply);
  return true;
}

bool DryRun() {
  const char* v = std::getenv("AMDSMI_DRY_RUN");
  return v != nullptr && std::strcmp(v, "1") == 0;
}

// Open a *private* system-bus connection so we never disturb a host
// application's shared bus.
DBusConnection* OpenSystemBus(const Dbus& d) {
  DbusError err;
  d.error_init(&err);
  DBusConnection* conn = d.bus_get_private(kBusSystem, &err);
  if (d.error_is_set(&err)) {
    d.error_free(&err);
    return nullptr;
  }
  if (conn != nullptr) d.conn_set_exit_on_disconnect(conn, 0);
  return conn;
}

void CloseBus(const Dbus& d, DBusConnection* conn) {
  if (conn == nullptr) return;
  d.conn_close(conn);
  d.conn_unref(conn);
}

}  // namespace

namespace amd {
namespace smi {
namespace detail {

// Select the carveout setting from a parsed list, preferring AMD's canonical
// attribute id over HP's UEFI-HII naming.
const BiosSetting* FindCarveout(const std::vector<BiosSetting>& settings) {
  static const char* const kCarveoutIds[] = {"com.amd-gpu.uma_carveout",
                                             "com.hp-bioscfg.Dedicated_Graphics_Memory"};
  for (const char* want : kCarveoutIds) {
    for (const auto& s : settings) {
      if (strcasecmp(s.id.c_str(), want) == 0) return &s;
    }
  }
  return nullptr;
}

// Map a resolved carveout setting into the public info struct: index the option
// descriptions (truncated to fit), clamp the count to the array bound, and set
// current_index to the matching option or to num_options ("unknown") when the
// current value is empty (e.g. redacted for an unprivileged reader).
amdsmi_status_t PopulateCarveoutInfo(const BiosSetting& setting, amdsmi_uma_carveout_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  if (setting.values.empty()) return AMDSMI_STATUS_NOT_SUPPORTED;
  uint32_t count = static_cast<uint32_t>(setting.values.size());
  if (count > AMDSMI_MAX_CARVEOUT_OPTIONS) count = AMDSMI_MAX_CARVEOUT_OPTIONS;
  uint32_t current = count;  // sentinel: "unknown" == num_options
  for (uint32_t i = 0; i < count; ++i) {
    info->options[i].index = i;
    std::strncpy(info->options[i].description, setting.values[i].c_str(),
                 AMDSMI_MAX_STRING_LENGTH - 1);
    info->options[i].description[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
    if (!setting.current.empty() && setting.current == setting.values[i]) current = i;
  }
  info->num_options = count;
  info->current_index = current;
  return AMDSMI_STATUS_SUCCESS;
}

// Validate a requested write against a resolved carveout setting (pure; no D-Bus).
amdsmi_status_t ValidateCarveoutWrite(const BiosSetting& setting, uint32_t option_index) {
  if (setting.read_only) return AMDSMI_STATUS_NO_PERM;
  if (option_index >= setting.values.size()) return AMDSMI_STATUS_INVAL;
  if (setting.name.empty()) return AMDSMI_STATUS_NOT_SUPPORTED;
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace detail

amdsmi_status_t fwupd_get_carveout_info(amdsmi_uma_carveout_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  const Dbus* d = LoadDbus();
  if (d == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;
  DBusConnection* conn = OpenSystemBus(*d);
  if (conn == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;

  std::vector<BiosSetting> settings;
  const bool ok = GetSettings(*d, conn, &settings);
  CloseBus(*d, conn);
  if (!ok) return AMDSMI_STATUS_NOT_SUPPORTED;

  const BiosSetting* c = detail::FindCarveout(settings);
  if (c == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;
  return detail::PopulateCarveoutInfo(*c, info);
}

amdsmi_status_t fwupd_set_carveout(uint32_t option_index) {
  const Dbus* d = LoadDbus();
  if (d == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;
  DBusConnection* conn = OpenSystemBus(*d);
  if (conn == nullptr) return AMDSMI_STATUS_NOT_SUPPORTED;

  std::vector<BiosSetting> settings;
  if (!GetSettings(*d, conn, &settings)) {
    CloseBus(*d, conn);
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  const BiosSetting* c = detail::FindCarveout(settings);
  if (c == nullptr) {
    CloseBus(*d, conn);
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  const amdsmi_status_t valid = detail::ValidateCarveoutWrite(*c, option_index);
  if (valid != AMDSMI_STATUS_SUCCESS) {
    CloseBus(*d, conn);
    return valid;
  }
  const std::string name = c->name;
  const std::string value = c->values[option_index];
  if (DryRun()) {
    CloseBus(*d, conn);
    return AMDSMI_STATUS_SUCCESS;
  }

  DbusError err;
  d->error_init(&err);
  DBusMessage* msg = d->msg_new_call(kService, kPath, kInterface, "SetBiosSettings");
  if (msg == nullptr) {
    CloseBus(*d, conn);
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  DbusIter it;
  d->iter_init_append(msg, &it);
  DbusIter arr;
  DbusIter ent;
  const char* k = name.c_str();
  const char* v = value.c_str();
  // Message construction can fail on OOM (each step returns FALSE); bail cleanly
  // rather than sending a malformed request.
  const bool built = d->iter_open(&it, kTypeArray, "{ss}", &arr) &&
                     d->iter_open(&arr, kTypeDictEntry, nullptr, &ent) &&
                     d->iter_append(&ent, kTypeString, &k) &&
                     d->iter_append(&ent, kTypeString, &v) && d->iter_close(&arr, &ent) &&
                     d->iter_close(&it, &arr);
  if (!built) {
    d->msg_unref(msg);
    CloseBus(*d, conn);
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  DBusMessage* reply = d->send_block(conn, msg, 30000, &err);
  d->msg_unref(msg);

  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  if (d->error_is_set(&err)) {
    const std::string ename = err.name ? err.name : "";
    const std::string emsg = err.message ? err.message : "";
    // Classify by the canonical D-Bus error name first (stable across fwupd
    // versions and locales); the human-readable message substrings are only a
    // best-effort fallback. fwupd returns NothingToDo when the value already
    // matches -- an idempotent success for our purposes.
    if (ename.find("NothingToDo") != std::string::npos ||
        emsg.find("already set") != std::string::npos ||
        emsg.find("no BIOS settings needed") != std::string::npos) {
      status = AMDSMI_STATUS_SUCCESS;
    } else if (ename.find("AccessDenied") != std::string::npos ||
               ename.find("AuthFailed") != std::string::npos ||
               ename.find("PermissionDenied") != std::string::npos ||
               emsg.find("not authorized") != std::string::npos ||
               emsg.find("permission") != std::string::npos) {
      status = AMDSMI_STATUS_NO_PERM;
    } else {
      status = AMDSMI_STATUS_NOT_SUPPORTED;
    }
    d->error_free(&err);
  } else if (reply == nullptr) {
    // No error reported but also no reply: treat as failure, not success.
    status = AMDSMI_STATUS_NOT_SUPPORTED;
  }
  if (reply != nullptr) d->msg_unref(reply);
  CloseBus(*d, conn);
  return status;
}

}  // namespace smi
}  // namespace amd
