################################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
################################################################################

import os
import os.path
import sys
import re
import yaml
import CppHeaderParser


# Numeric field type (abstract).
class _NumericFt:
    # Returns the C++ expression to cast the expression `expr` to the C
    # type of this field type.
    def cast(self, expr):
        return f"static_cast<{self.c_type}>({expr})"


# Integer field type (abstract).
class _IntFt(_NumericFt):
    def __init__(self, size, pref_disp_base="dec"):
        self._size = size
        self._pref_disp_base = pref_disp_base

    # Size (bits).
    @property
    def size(self):
        return self._size

    # Preferred display base (`dec` or `hex`).
    @property
    def pref_disp_base(self):
        return self._pref_disp_base

    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        return {
            "size": self._size,
            "preferred-display-base": self._pref_disp_base,
        }


# Signed integer field type.
class _SIntFt(_IntFt):
    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        ret = super().barectf_yaml
        ret["class"] = "sint"
        return ret

    # Equivalent C type
    @property
    def c_type(self):
        return f"std::int{self._size}_t"


# Unsigned integer field type.
class _UIntFt(_IntFt):
    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        ret = super().barectf_yaml
        ret["class"] = "uint"
        return ret

    # Equivalent C type.
    @property
    def c_type(self):
        return f"std::uint{self._size}_t"


# Pointer field type.
class _PointerFt(_UIntFt):
    def __init__(self):
        super().__init__(64, "hex")

    # Returns the C++ expression to cast the expression `expr` to the C
    # type of this field type.
    def cast(self, expr):
        return f"static_cast<{self.c_type}>(reinterpret_cast<std::uintptr_t>({expr}))"


# Enumeration field type (abstract).
class _EnumFt(_IntFt):
    def __init__(self, size, mappings):
        super().__init__(size)
        self._mappings = mappings.copy()

    # Mappings (names to integers).
    @property
    def mappings(self):
        return self._mappings

    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        ret = super().barectf_yaml
        mappings = {}

        for name, val in self._mappings.items():
            mappings[name] = [val]

        ret["mappings"] = mappings
        return ret


# Unsigned enumeration field type.
class _UEnumFt(_EnumFt, _UIntFt):
    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        ret = super().barectf_yaml
        ret["class"] = "uenum"
        return ret


# Signed enumeration field type.
class _SEnumFt(_EnumFt, _UIntFt):
    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        ret = super().barectf_yaml
        ret["class"] = "senum"
        return ret


# Optional string field type.
class _OptStrFt:
    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        return {
            "class": "str",
        }


# String field type.
class _StrFt(_OptStrFt):
    pass


# Floating-point number field type.
class _FloatFt(_NumericFt):
    def __init__(self, size):
        self._size = size

    # Size (bits): 32 or 64.
    @property
    def size(self):
        return self._size

    # Equivalent barectf field type in YAML.
    @property
    def barectf_yaml(self):
        return {
            "class": "real",
            "size": self._size,
        }

    # Equivalent C type.
    @property
    def c_type(self):
        if self._size == 32:
            return "float"
        else:
            assert self._size == 64
            return "double"


# Event record type.
class _Ert:
    def __init__(self, api_func_name, members):
        self._api_func_name = api_func_name
        self._members = members

    # API function name
    @property
    def api_func_name(self):
        return self._api_func_name

    # Parameters of function (list of `_ErtMember`).
    @property
    def members(self):
        return self._members


# Beginning event record type.
class _BeginErt(_Ert):
    # Name of event record type depending on the API prefix.
    def name(self, api_prefix):
        suffix = "_begin" if api_prefix == "hsa" else "Begin"
        return f"{self._api_func_name}{suffix}"


# End event record type.
class _EndErt(_Ert):
    # Name of event record type depending on the API prefix.
    def name(self, api_prefix):
        suffix = "_end" if api_prefix == "hsa" else "End"
        return f"{self._api_func_name}{suffix}"


# Event record type member.
class _ErtMember:
    def __init__(self, access, member_names, ft):
        self._access = access
        self._member_names = member_names.copy()
        self._ft = ft

    # C++ access expression.
    @property
    def access(self):
        return self._access

    # List of member names.
    @property
    def member_names(self):
        return self._member_names

    # Equivalent field type.
    @property
    def ft(self):
        return self._ft


# Makes sure some condition is satisfied, or prints the error message
# `error_msg` and quits with exit status 1 otherwise.
#
# This is an unconditional assertion.
def _make_sure(cond, error_msg):
    if not cond:
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)


def _enumerator_effective_val(enum_val):
    # Try the value, but this value may be a string (an
    # enumerator/definition).
    val = enum_val.get("value")

    if type(val) is int:
        return val

    # Try the raw value.
    val = enum_val.get("raw_value")

    if val is not None:
        if type(val) is int:
            # Raw value is already an integer.
            return val
        else:
            # Try to parse the raw value string as an integer.
            try:
                return int(val, 0)
            except:
                pass

    _make_sure(False, f'Cannot get the integral value of enumerator `{enum_val["name"]}`')


# Returns the equivalent field type of the C type `c_type`.
def _number_ft_from_c_type(cpp_header, c_type):
    # Check for known enumeration.
    m = re.match(r"(?:enum\s+)?(\w+)", c_type)

    if m:
        size = 32

        for enum_info in cpp_header.enums:
            if m.group(1) == enum_info.get("name"):
                # Fill enumeration field type mappings.
                mappings = {
                    str(v["name"]): _enumerator_effective_val(v)
                    for v in enum_info["values"]
                }

                if len(mappings) == 0:
                    return _SIntFt(64)

                if max(mappings.values()) >= 2**31 or min(mappings.values()) < -(
                    2**31
                ):
                    size = 64

                _make_sure(
                    len(mappings) > 0, f'Enumeration `{enum_info["name"]}` is empty'
                )

                # Create corresponding enumeration field type.
                return _SEnumFt(size, mappings)

    # Find corresponding basic field type.
    is_unsigned = "unsigned" in c_type

    if "long" in c_type:
        if is_unsigned:
            return _UIntFt(64)
        else:
            return _SIntFt(64)
    elif "short" in c_type:
        if is_unsigned:
            return _UIntFt(16)
        else:
            return _SIntFt(16)
    elif "char" in c_type:
        if is_unsigned:
            return _UIntFt(8)
        else:
            return _SIntFt(8)
    elif "float" in c_type:
        return _FloatFt(32)
    elif "double" in c_type:
        return _FloatFt(64)
    else:
        # Assume `int` (often an unresolved C enumeration).
        if is_unsigned:
            return _UIntFt(32)
        else:
            return _SIntFt(32)


# Returns whether or not a property has a pointer type.
def _prop_is_pointer(prop, c_type):
    if prop["pointer"] or prop["function_pointer"]:
        return True

    if prop["array"] and "array_size" in prop:
        return True

    if prop["unresolved"]:
        # HSA API function pointers.
        if prop["name"] in ("callback", "handler"):
            return True

        # HIP API function pointers.
        if c_type.endswith("Fn_t"):
            return True

    # Check the C type itself.
    if "*" in c_type or "*" in prop.get("raw_type", ""):
        return True

    return False


# Returns a list of event record type member objects for the structure
# `struct` considering the initial C++ access expression `access` and
# member names `member_names`.
def _get_ert_members_for_struct(cpp_header, struct, access, member_names):
    members = []
    member_names = member_names.copy()
    member_names.append(None)
    props = struct["properties"]["public"]

    for index, prop in enumerate(props):
        # Property name.
        name = prop["name"]

        # Member names, access, and C type.
        member_names[-1] = str(name)
        this_access = f"{access}.{name}"
        c_type = prop["type"]
        aliases = prop["aliases"]

        # Skip no type.
        if c_type == "":
            continue

        # Skip unnamed or union.
        if name == "" or "union" in name or re.match(r"\bunion\b", c_type):
            continue

        # Check for known C type alias.
        while True:
            c_type_alias = cpp_header.typedefs.get(c_type)

            if c_type_alias is None:
                break

            c_type = c_type_alias

        # Check for C string.
        if re.match(r"^((const\s+char)|(char\s+const)|char)\s*\*$", c_type.strip()):
            members.append(_ErtMember(this_access, member_names, _OptStrFt()))
            continue

        # Check for pointer.
        if _prop_is_pointer(prop, c_type):
            # Pointer: use numeric value.
            members.append(_ErtMember(this_access, member_names, _PointerFt()))
            continue

        # Check for substructure.
        sub_struct = cpp_header.classes.get(c_type)

        if sub_struct is None and len(aliases) == 1:
            sub_struct = cpp_header.classes.get(aliases[0])

        if sub_struct is not None:
            members += _get_ert_members_for_struct(
                cpp_header, sub_struct, this_access, member_names
            )
            continue

        # Use a basic field type.
        members.append(
            _ErtMember(
                this_access, member_names, _number_ft_from_c_type(cpp_header, c_type)
            )
        )

    return members


# Returns the beginning and end event record type objects for the
# callback data structure `struct`.
def _erts_from_cb_data_struct(api_prefix, cpp_header, retval_info, struct):
    # The location of the `args` union within the nested structures of
    # `struct`.
    args_nested_cls_index = 0

    # Create return value members (to be used later).
    if retval_info is not None:
        args_nested_cls_index = 1
        retval_members = {}
        nested_classes = struct["nested_classes"]
        _make_sure(
            len(nested_classes) >= 1,
            f"Return value union doesn't exist in `{struct['name']}`",
        )
        retval_union = nested_classes[0]

        for prop in retval_union["properties"]["public"]:
            name = str(prop["name"])
            member = _ErtMember(
                f"GetApiData().{name}",
                ["retval"],
                _number_ft_from_c_type(cpp_header, prop["type"]),
            )
            retval_members[prop["name"]] = member

        # Make sure we have everything we need.
        for api_func_name, retval_name in retval_info.items():
            if retval_name is not None:
                _make_sure(
                    retval_name in retval_members,
                    f"Return value union member `{retval_name}` doesn't exist (function {api_func_name}())",
                )

    # Create beginning/end event record type objects.
    begin_erts = []
    end_erts = []
    nested_classes = struct["nested_classes"][args_nested_cls_index]["nested_classes"]
    props = struct["nested_classes"][args_nested_cls_index]["properties"]["public"]
    _make_sure(
        len(nested_classes) == len(props),
        f'Mismatch between nested structure and member count in `{struct["name"]}`',
    )

    for index, prop in enumerate(props):
        # API function name is the name of the member.
        api_func_name = str(prop["name"])

        # Get the parameters.
        members = _get_ert_members_for_struct(
            cpp_header, nested_classes[index], f"GetApiData().args.{api_func_name}", []
        )

        # Append new beginning event record type object.
        begin_erts.append(_BeginErt(api_func_name, members))

        # Append new end event record type object if possible.
        ret_members = []

        if retval_info is not None:
            retval_type = retval_info.get(api_func_name)

            if retval_type is not None:
                ret_members.append(retval_members[retval_type])

        end_erts.append(_EndErt(api_func_name, ret_members))

    return begin_erts, end_erts


# Creates and returns the return value information dictionary.
#
# This dictionary maps API function names to the member to get within
# the callback data structure.
#
# This only applies to the HSA API: for other APIs, this function
# returns `None`.
def _get_retval_info(path):
    if "hsa" not in os.path.basename(path):
        return

    retval_info = {}
    cur_api_func_name = None

    with open(path) as f:
        for line in f:
            if 'out << ")' in line and cur_api_func_name is not None:
                m = re.search(r"api_data.(\w+_retval)", line)
                retval_info[cur_api_func_name] = m.group(1) if m else None
            else:
                m = re.search(r'out << "(hsa_\w+)\(";', line)

                if m:
                    cur_api_func_name = m.group(1)

    return retval_info


# Returns a partial barectf data stream type in YAML with the event
# record types `erts`.
def _yaml_dst_from_erts(api_prefix, erts):
    # Base.
    yaml_erts = {}
    yaml_dst = {
        "event-record-types": yaml_erts,
    }

    # Create one event record type per API function.
    for ert in erts:
        # Base.
        yaml_members = []
        yaml_ert = {
            "payload-field-type": {
                "class": "struct",
                "members": yaml_members,
            },
        }

        # Create one structure field type member per member.
        for member in ert.members:
            # barectf doesn't support nested CTF structures, so join
            # individual member names with `__` to flatten.
            yaml_members.append(
                {
                    "_"
                    + "__".join(member.member_names): {
                        "field-type": member.ft.barectf_yaml,
                    },
                }
            )

        # Add event record type.
        yaml_erts[ert.name(api_prefix)] = yaml_ert

    # Convert to YAML.
    return yaml.dump(yaml_dst)


# Returns the C++ switch statement which calls the correct barectf
# tracing function depending on the API function operation ID.
def _cpp_switch_statement_from_erts(api_prefix, erts):
    lines = []
    lines.append("switch (GetOp()) {")

    for ert in erts:
        if api_prefix == 'hip' and 'R0600' in ert.api_func_name:
            continue
        lines.append(f"  case {api_prefix.upper()}_API_ID_{ert.api_func_name}:")
        lines.append(f"    barectf_{api_prefix}_api_trace_{ert.name(api_prefix)}(")
        lines.append(f"      &barectf_ctx,")
        lines.append(f"      GetThreadId(),")
        lines.append(f"      GetQueueId(),")
        lines.append(f"      GetAgentId(),")
        lines.append(f"      GetCorrelationId(),")

        if api_prefix == "hip":
            lines.append(f"      GetKernelName().c_str(),")

        if len(ert.members) == 0:
            # Remove last comma.
            lines[-1] = lines[-1].replace(",", "")

        for index, member in enumerate(ert.members):
            if type(member.ft) is _OptStrFt:
                # Only dereference C string if not null, otherwise use
                # an empty string.
                lines.append(f'      {member.access} ? {member.access} : ""')
            elif type(member.ft) is _StrFt:
                lines.append(f"      {member.access}")
            else:
                lines.append(f"      {member.ft.cast(member.access)}")

            if index + 1 < len(ert.members):
                lines[-1] += ","

        lines.append("    );")
        lines.append("    break;")

    lines.append("}")
    return lines


# Returns a set of expected API function names based on the
# enumerators of the `*_api_id_t` enumeration.
def _api_func_names(api_prefix, cpp_header):
    # Find the `*_api_id_t` enumeration.
    for enum in cpp_header.enums:
        if enum.get('name') == f'{api_prefix}_api_id_t':
            break

    # Create the set of API function names based on enumerators.
    func_names = set()
    pat = re.compile(rf'{api_prefix.upper()}_API_ID_(_*{api_prefix}.+)$')

    for entry in enum['values']:
        if type(entry['value']) is str and 'API_ID_NONE' in entry['value']:
            # An enumerator may have the value `*_API_ID_NONE` which
            # means the corresponding API function is not available.
            continue

        m = pat.match(entry['name'])

        if m is not None:
            func_names.add(m.group(1))

    # Return API function names
    return func_names


# Processes the complete API header file `path`.
def _process_file(api_prefix, path):
    # Create `CppHeader` object.
    try:
        cpp_header = CppHeaderParser.CppHeader(path)
    except CppHeaderParser.CppParseError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    # Get return value information dictionary.
    retval_info = _get_retval_info(path)

    # add support for structures defined inside union.
    new_items = []
    for struct_name, struct in cpp_header.classes.items():
        # Check if the struct_name starts with 'union '.
        if re.match(r'^union \w+', struct_name) is not None:
            parts = struct_name.split('::')
            simplified = parts[-1] 
            if simplified != "union ":
                new_items.append((simplified, cpp_header.classes.get(struct_name)))

    for key, value in new_items:
        cpp_header.classes[key] = value

    # Find callback data structure.
    for struct_name, struct in cpp_header.classes.items():
        if re.match(r'^' + api_prefix + r'_api_data\w+$', struct_name) is not None:
            break

    # Process callback data structure.
    begin_erts, end_erts = _erts_from_cb_data_struct(api_prefix,
                                                     cpp_header,
                                                     retval_info,
                                                     struct)

    # API functions without parameters are not part of the callback data
    # structure, but they have an ID in the `*_api_id_t` enumeration.
    #
    # Add missing event record types to `begin_erts` and `end_erts`
    # considering the `*_api_id_t` enumeration.
    processed_api_func_names = set([ert.api_func_name for ert in begin_erts])

    for func_name in _api_func_names(api_prefix, cpp_header):
        if func_name not in processed_api_func_names:
            begin_erts.append(_BeginErt(func_name, []))
            end_erts.append(_EndErt(func_name, []))

    # Write barectf YAML file.
    with open(f'{api_prefix}_erts.yaml', 'w') as f:
        f.write(_yaml_dst_from_erts(api_prefix, begin_erts + end_erts))

    # Write C++ code (beginning event record).
    with open(f'{api_prefix}_begin.cpp.i', 'w') as f:
        f.write('\n'.join(_cpp_switch_statement_from_erts(api_prefix,
                                                          begin_erts)))

    # Write C++ code (end event record).
    with open(f'{api_prefix}_end.cpp.i', 'w') as f:
        f.write('\n'.join(_cpp_switch_statement_from_erts(api_prefix,
                                                          end_erts)))


if __name__ == "__main__":
    # Disable `CppHeaderParser` printing to standard output.
    CppHeaderParser.CppHeaderParser.print_warnings = 0
    CppHeaderParser.CppHeaderParser.print_errors = 0
    CppHeaderParser.CppHeaderParser.debug = 0
    CppHeaderParser.CppHeaderParser.debug_trace = 0

    # Process the complete API header file.
    _process_file(sys.argv[1], sys.argv[2])
