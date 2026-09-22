################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

import csv, sqlite3, re, sys
from functools import reduce
from txt2params import gen_params


# SQLite Database class
class SQLiteDB:
    def __init__(self, file_name):
        self.connection = sqlite3.connect(file_name)
        self.tables = {}
        self.section_index = 0

    def __del__(self):
        self.connection.close()

    # add DB table
    def add_table(self, name, descr, extra=()):
        (field_list, field_dict) = descr
        if name in self.tables:
            raise Exception('table is already added: "' + name + '"')

        # create DB table
        table_descr = []
        for field in field_list:
            table_descr.append('"%s" %s' % (field, field_dict[field]))
        for item in extra:
            table_descr.append('"%s" %s' % (item[0], item[1]))
        stm = "CREATE TABLE " + name + " (%s)" % ", ".join(table_descr)
        cursor = self.connection.cursor()
        cursor.execute(stm)
        self.connection.commit()

        # register table
        fields_str = ",".join(map(lambda x: '"' + x + '"', field_list))
        templ_str = ",".join("?" * len(field_list))
        stm = "INSERT INTO " + name + "(" + fields_str + ") VALUES(" + templ_str + ");"
        self.tables[name] = stm

        return (cursor, stm)

    # add columns to table
    def add_columns(self, name, columns):
        cursor = self.connection.cursor()
        for item in columns:
            stm = "ALTER TABLE " + name + ' ADD COLUMN "%s" %s' % (item[0], item[1])
            cursor.execute(stm)
        self.connection.commit()

    # add columns with expression
    def add_data_column(self, table_name, data_label, data_type, data_expr):
        cursor = self.connection.cursor()
        cursor.execute(
            'ALTER TABLE %s ADD COLUMN "%s" %s' % (table_name, data_label, data_type)
        )
        cursor.execute("UPDATE %s SET %s = (%s);" % (table_name, data_label, data_expr))

    def change_rec_name(self, table_name, rec_id, rec_name):
        self.connection.execute(
            "UPDATE " + table_name + ' SET Name = ? WHERE "Index" = ?', (rec_name, rec_id)
        )

    def change_rec_tid(self, table_name, rec_id, tid):
        self.connection.execute(
            "UPDATE " + table_name + ' SET tid = ? WHERE "Index" = ?', (tid, rec_id)
        )

    def change_rec_fld(self, table_name, fld_expr, rec_pat):
        self.connection.execute(
            "UPDATE " + table_name + " SET " + fld_expr + " WHERE " + rec_pat
        )

    def table_get_record(self, table_name, rec_pat):
        cursor = self.connection.execute(
            "SELECT * FROM " + table_name + " WHERE " + rec_pat
        )
        raws = cursor.fetchall()
        if len(raws) != 1:
            raise Exception(
                "Record (" + rec_pat + ') is not unique, table "' + table_name + '"'
            )
        return list(raws[0])

    # populate DB table entry
    def insert_entry(self, table, val_list):
        (cursor, stm) = table
        cursor.execute(stm, val_list)

    # populate DB table entry
    def commit_entry(self, table, val_list):
        self.insert_entry(table, val_list)
        self.connection.commit()

    # populate DB table data
    def insert_table(self, table, reader):
        for val_list in reader:
            if not val_list[-1]:
                val_list.pop()
            self.insert_entry(table, val_list)
        self.connection.commit()

    # return table fields list
    def _get_fields(self, table_name):
        cursor = self.connection.execute("SELECT * FROM " + table_name)
        return list(map(lambda x: '"%s"' % (x[0]), cursor.description))

    # return table raws list
    def _get_raws(self, table_name):
        cursor = self.connection.execute("SELECT * FROM " + table_name)
        return cursor.fetchall()

    def _get_raws_indexed(self, table_name):
        cursor = self.connection.execute(
            "SELECT * FROM " + table_name + ' order by "Index" asc;'
        )
        return cursor.fetchall()

    def _get_raw_by_id(self, table_name, rec_id):
        cursor = self.connection.execute(
            "SELECT * FROM " + table_name + ' WHERE "Index"=?', (rec_id,)
        )
        raws = cursor.fetchall()
        if len(raws) != 1:
            raise Exception('Index is not unique, table "' + table_name + '"')
        return list(raws[0])

    def table_get_raws(self, table_name):
        return self._get_raws(table_name)

    # dump CSV table
    def dump_csv(self, table_name, file_name):
        if not re.search(r"\.csv$", file_name):
            raise Exception('wrong output file type: "' + file_name + '"')

        fields = self._get_fields(table_name)
        with open(file_name, mode="w") as fd:
            fd.write(",".join(fields) + "\n")
            for raw in self._get_raws(table_name):
                tmp = list(raw)
                for idx in range(len(tmp)):
                    if type(tmp[idx]) == str:
                        if not (tmp[idx][0] == tmp[idx][-1] == '"'):
                            tmp[idx] = '"' + tmp[idx] + '"'
                raw = tuple(tmp)
                fd.write(reduce(lambda a, b: str(a) + "," + str(b), raw) + "\n")

    # dump JSON trace
    def open_json(self, file_name):
        if not re.search(r"\.json$", file_name):
            raise Exception('wrong output file type: "' + file_name + '"')
        with open(file_name, mode="w") as fd:
            fd.write('{ "traceEvents":[{}\n')

    def close_json(self, file_name):
        if not re.search(r"\.json$", file_name):
            raise Exception('wrong output file type: "' + file_name + '"')
        with open(file_name, mode="a") as fd:
            fd.write("}")

    def label_json(self, pid, label, file_name):
        if not re.search(r"\.json$", file_name):
            raise Exception('wrong output file type: "' + file_name + '"')
        with open(file_name, mode="a") as fd:
            fd.write(
                ',{"args":{"name":"%s"},"ph":"M","pid":%s,"name":"process_name","sort_index":%d}\n'
                % (label, pid, self.section_index)
            )
        self.section_index += 1

    def flow_json(
        self, base_id, from_pid, from_us_list, to_pid, to_us_dict, corr_id_list, file_name
    ):
        if not re.search(r"\.json$", file_name):
            raise Exception('wrong output file type: "' + file_name + '"')
        with open(file_name, mode="a") as fd:
            dep_id = base_id
            for ind in range(len(from_us_list)):
                corr_id = corr_id_list[ind] if (len(corr_id_list) != 0) else ind
                if corr_id in to_us_dict:
                    (from_ts, stream_id, tid) = from_us_list[ind]
                    to_ts = to_us_dict[corr_id]
                    if from_ts > to_ts:
                        from_ts = to_ts
                    fd.write(
                        ',{"ts":%d,"ph":"s","cat":"DataFlow","id":%d,"pid":%d,"tid":%d,"name":"dep"}\n'
                        % (from_ts, dep_id, from_pid, tid)
                    )
                    fd.write(
                        ',{"ts":%d,"ph":"t","cat":"DataFlow","id":%d,"pid":%d,"tid":%d,"name":"dep"}\n'
                        % (to_ts, dep_id, to_pid, stream_id)
                    )
                    dep_id += 1

    def metadata_json(self, jsonfile, sysinfo_file):
        params = gen_params(sysinfo_file)
        with open(jsonfile, mode="a") as fd:
            cnt = 0
            fd.write("],\n")
            fd.write('"otherData": {\n')
            for nkey in sorted(params.keys()):
                key = nkey[1]
                cnt = cnt + 1
                if cnt == len(params):
                    fd.write('    "' + key + '": "' + params[nkey] + '"\n')
                else:
                    fd.write('    "' + key + '": "' + params[nkey] + '",\n')
            fd.write("  }\n")

    def dump_json(self, table_name, data_name, file_name):
        if not re.search(r"\.json$", file_name):
            raise Exception('wrong output file type: "' + file_name + '"')

        sub_ptrn = re.compile(r'(^"|"$)')
        name_ptrn = re.compile(r"(name|Name)")

        table_fields = self._get_fields(table_name)
        table_raws = self._get_raws(table_name)
        data_fields = self._get_fields(data_name)
        data_raws = self._get_raws(data_name)

        with open(file_name, mode="a") as fd:
            table_raws_len = len(table_raws)
            for raw_index in range(table_raws_len):
                if (raw_index == table_raws_len - 1) or (raw_index % 1000 == 0):
                    sys.stdout.write(
                        "\rdump json "
                        + str(raw_index)
                        + ":"
                        + str(len(table_raws))
                        + " " * 100
                    )

                vals_list = []
                values = list(table_raws[raw_index])
                for value_index in range(len(values)):
                    label = table_fields[value_index]
                    value = values[value_index]
                    if name_ptrn.search(label):
                        value = sub_ptrn.sub(r"", value)
                    if label != '"Index"':
                        if label == '"dur"' and value == 0:
                            vals_list.append('%s:"%s"' % (label, "1"))
                        elif label == '"pid"' or label == '"tid"':
                            vals_list.append('%s:%d' % (label, value))
                        else:
                            vals_list.append('%s:"%s"' % (label, value))

                args_list = []
                data = list(data_raws[raw_index])
                for value_index in range(len(data)):
                    label = data_fields[value_index]
                    value = data[value_index]
                    if label[:3] == '"__':
                        continue
                    if name_ptrn.search(label):
                        value = sub_ptrn.sub(r"", value)
                    if label != '"Index"' and label != '"roctx-range"':
                        args_list.append('%s:"%s"' % (label, value))

                fd.write(
                    ',{"ph":"%s",%s,\n  "args":{\n    %s\n  }\n}\n'
                    % ("X", ",".join(vals_list), ",\n    ".join(args_list))
                )

        sys.stdout.write("\n")

    # execute query on DB
    def execute(self, cmd):
        cursor = self.connection.cursor()
        cursor.execute(cmd)

    # commit DB
    def commit(self):
        self.connection.commit()

    # close DB
    def close(self):
        self.connection.commit()
        self.connection.close()

    # access DB
    def get_raws(self, table_name):
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM %s" % table_name)
        return cur.fetchall()

    # return CSV descriptor
    # list of fields and dictionaly for the fields types
    def _get_csv_descr(self, table_name, fd):
        reader = csv.DictReader(fd)
        field_names = reader.fieldnames
        if not field_names[-1]:
            field_names.pop()
        field_types = {}

        for entry in reader:
            fields_left = [f for f in field_names if f not in field_types.keys()]
            # all fields processed
            if not fields_left:
                break

            for field in fields_left:
                data = entry[field]
                # need data for the field to be processed
                if len(data) == 0:
                    continue

                if data.isdigit():
                    field_types[field] = "INTEGER"
                else:
                    field_types[field] = "TEXT"

        if len(fields_left) > 0:
            raise Exception("types not found for fields: ", fields_left)
        return (field_names, field_types)

    # add CSV table
    def add_csv_table(self, table_name, file_name, extra=()):
        with open(file_name, mode="r") as fd:
            # get CSV table descriptor
            descr = self._get_csv_descr(table_name, fd)
            # reader to populate the table
            fd.seek(0)
            reader = csv.reader(fd)
            reader.next()
            table = self.add_table(table_name, descr, extra)
            self.insert_table(table, reader)


##############################################################################################
