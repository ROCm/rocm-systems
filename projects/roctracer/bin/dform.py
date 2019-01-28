#!/usr/bin/python
from sqlitedb import SQLiteDB

def post_process_data(db, table_name, outfile = ''): 
#  db.add_data_column('A', 'DispDurNs', 'INTEGER', 'BeginNs - DispatchNs')
#  db.add_data_column('A', 'ComplDurNs', 'INTEGER', 'CompleteNs - EndNs')
#  db.add_data_column('A', 'TotalDurNs', 'INTEGER', 'CompleteNs - DispatchNs')
#  db.add_data_column('A', 'TimeNs', 'INTEGER', 'BeginNs - (select BeginNs from A limit 1)')
  db.add_data_column(table_name, 'DurationNs', 'INTEGER', 'EndNs - BeginNs')
  if outfile != '': db.dump_csv(table_name, outfile)

def gen_data_bins(db, outfile):
  db.execute('create view C as select Name, Calls, TotalDurationNs, TotalDurationNs/Calls as AverageNs, TotalDurationNs*100.0/(select sum(TotalDurationNs) from %s) as Percentage from %s order by TotalDurationNs desc;' % ('B', 'B'));
  db.dump_csv('C', outfile)
  db.execute('DROP VIEW C')

def gen_table_bins(db, table, outfile, name_var, dur_ns_var):
  db.execute('create view B as select (%s) as Name, count(%s) as Calls, sum(%s) as TotalDurationNs from %s group by %s' % (name_var, name_var, dur_ns_var, table, name_var))
  gen_data_bins(db, outfile)
  db.execute('DROP VIEW B')

def gen_api_json_trace(db, table, outfile):
  db.execute('create view B as select Name as name, (BeginNs/1000) as ts, (DurationNs/1000) as dur from %s order by ts asc;' % table);
  db.dump_json('B', table, outfile)
  db.execute('DROP VIEW B')

def gen_kernel_json_trace(db, table, outfile):
  db.execute('create view B as select (KernelName) as name, ("gpu-id") as pid, (0) as tid, (BeginNs/1000) as ts, (DurationNs/1000) as dur from %s order by ts asc;' % table);
  db.dump_json('B', table, outfile)
  db.execute('DROP VIEW B')
##############################################################################################
