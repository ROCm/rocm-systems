#include "shared/include/utils.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace wsl {
namespace thunk {

namespace {
static const struct GfxipTable kGfxipTable[] = {
  { 0x7448, 11, 0, 0 },
  { 0x744C, 11, 0, 0 },
  { 0x745E, 11, 0, 0 },
  { 0x7449, 11, 0, 0 },
  { 0x744a, 11, 0, 0 },
  { 0x744b, 11, 0, 0 },
  { 0x7470, 11, 0, 1 },
  { 0x747E, 11, 0, 1 },
  { 0x7590, 12, 0, 0 },
  { 0x7550, 12, 0, 1 },
  { 0x7551, 12, 0, 1 },
  { 0x150E, 11, 5, 0 },
  { 0x1586, 11, 5, 1 },
  { 0x1114, 11, 5, 2 },
  { 0x1900, 11, 0, 3 },
  { 0x1902, 11, 5, 3 },
};

const int kGfxipTableSize = sizeof(kGfxipTable) / sizeof(kGfxipTable[0]);

constexpr size_t kMaxExtraGfxipEntries = 256;
constexpr char kDidsRelativePath[] = "/share/rocdxg/dids.conf";

// Anchor in this TU for dladdr (resolves to the loaded librocdxg.so).
void DidsConfPathAnchor() {}

std::string GetDidsConfPath() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(&DidsConfPathAnchor), &info) == 0 ||
      info.dli_fname == nullptr)
    return {};

  const std::string lib_path(info.dli_fname);
  const auto slash = lib_path.rfind('/');
  if (slash == std::string::npos)
    return {};

  const std::string lib_dir = lib_path.substr(0, slash);
  const auto parent_slash = lib_dir.rfind('/');
  if (parent_slash == std::string::npos)
    return {};

  return lib_dir.substr(0, parent_slash) + kDidsRelativePath;
}

unsigned long ParseDecimalOrHex(const char *line, char **end) {
  if (line[0] == '0' && (line[1] == 'x' || line[1] == 'X'))
    return strtoul(line, end, 16);
  return strtoul(line, end, 10);
}

bool ParseGfxipLine(const char *line, GfxipTable *out) {
  while (*line == ' ' || *line == '\t')
    ++line;
  if (*line == '\0' || *line == '#')
    return false;

  char *end = nullptr;
  const unsigned long device_id = ParseDecimalOrHex(line, &end);
  if (end == line || *end != ',')
    return false;

  line = end + 1;
  const unsigned long major = ParseDecimalOrHex(line, &end);
  if (end == line || *end != ',')
    return false;

  line = end + 1;
  const unsigned long minor = ParseDecimalOrHex(line, &end);
  if (end == line || *end != ',')
    return false;

  line = end + 1;
  const unsigned long stepping = ParseDecimalOrHex(line, &end);
  if (end == line)
    return false;

  if (device_id > 0xFFFF || major > 255 || major == 0 || minor > 255 || stepping > 255)
    return false;

  out->device_id = static_cast<uint16_t>(device_id);
  out->major = static_cast<uint8_t>(major);
  out->minor = static_cast<uint8_t>(minor);
  out->stepping = static_cast<uint8_t>(stepping);
  return true;
}

void LoadUserDidsFromFile(const std::string &path,
                          std::vector<GfxipTable> *out) {
  if (path.empty())
    return;

  std::ifstream in(path);
  if (!in)
    return;

  std::string line;
  while (std::getline(in, line) && out->size() < kMaxExtraGfxipEntries) {
    // Strip trailing CR/whitespace so CRLF-edited files and stray spaces
    // don't depend on the parser's lenient last-field handling.
    while (!line.empty() &&
           (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
      line.pop_back();

    GfxipTable entry{};
    if (ParseGfxipLine(line.c_str(), &entry))
      out->push_back(entry);
  }
}

const GfxipTable *FindGfxipEntry(const std::vector<GfxipTable> &table,
                                 uint16_t device_id) {
  for (const auto &entry : table) {
    if (entry.device_id == device_id)
      return &entry;
  }
  return nullptr;
}

std::vector<GfxipTable> g_merged_gfxip;
std::once_flag g_merged_gfxip_once;

void InitMergedGfxip() {
  g_merged_gfxip.assign(kGfxipTable, kGfxipTable + kGfxipTableSize);

  // To disable loading user-supplied IDs from dids.conf, return here.
  // if (std::getenv("ROCDXG_DISABLE_DIDS_CONF"))
  //   return;

  std::vector<GfxipTable> extra;
  LoadUserDidsFromFile(GetDidsConfPath(), &extra);

  for (const auto &entry : extra) {
    if (FindGfxipEntry(g_merged_gfxip, entry.device_id) == nullptr)
      g_merged_gfxip.push_back(entry);
  }
}

void EnsureMergedGfxipLoaded() {
  std::call_once(g_merged_gfxip_once, InitMergedGfxip);
}

} // namespace

bool QueryAdapterSupported(unsigned int device_id) {
  EnsureMergedGfxipLoaded();
  return FindGfxipEntry(g_merged_gfxip,
                        static_cast<uint16_t>(device_id)) != nullptr;
}

bool LookupGfxipEntry(uint16_t device_id, GfxipTable *out) {
  if (out == nullptr)
    return false;

  EnsureMergedGfxipLoaded();
  const GfxipTable *entry = FindGfxipEntry(g_merged_gfxip, device_id);
  if (entry == nullptr)
    return false;

  *out = *entry;
  return true;
}

} // namespace thunk
} // namespace wsl
