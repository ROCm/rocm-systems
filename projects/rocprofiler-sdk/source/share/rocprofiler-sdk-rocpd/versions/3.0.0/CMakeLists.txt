#
#
#

set(rocpd_schemas data_views.sql rocpd_indexes.sql rocpd_metadata.sql rocpd_tables.sql
                  rocpd_views.sql summary_views.sql)

rocpd_configure_schema_files(VERSION "3.0.0" FILES ${rocpd_schemas})
