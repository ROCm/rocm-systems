// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_db.h"
#include "sqlite3.h" 
#include <set>
#include <mutex>
#include <condition_variable>
#include "rocprofvis_db_query_builder.h"
#include <array>

namespace RocProfVis
{
namespace DataModel
{

#define MAX_CONNECTIONS 100

// type of sqlite3_exec callback function
typedef int (*RpvSqliteExecuteQueryCallback)(void*, int, sqlite3_stmt*, char**);
typedef struct SQLInsertParams
{
    const char* column;
    const char* type;
} SQLInsertParams;

typedef enum rocprofvis_db_sqlite_query_type_t
{
    kRPVSourceQueryGeneric = 0,
    kRPVSourceQueryTrackByQueue = 0,
    kRPVSourceQueryTrackByStream = 1,
    kRPVCacheTableName = 1,
    kRPVSourceQueryLevel = 2,
    kRPVSourceQuerySliceByQueue = 3,
    kRPVSourceQuerySliceByStream = 4,
    kRPVSourceQueryTable = 5,
    kRPVNumSourceQueryTypes = 6
} rocprofvis_db_sqlite_query_type_t;

typedef enum rocprofvis_dm_track_search_id_t
{
    kRPVTrackSearchIdThreads,
    kRPVTrackSearchIdThreadSamples,
    kRPVTrackSearchIdDispatches,
    kRPVTrackSearchIdMemAllocs,
    kRPVTrackSearchIdMemCopies,
    kRPVTrackSearchIdCounters,
    kRPVTrackSearchIdStreams,
    kRPVTrackSearchIdUnknown,
} rocprofvis_dm_track_search_id_t;

// structure to pass parameters to sqlite3_exec callbacks
typedef struct{
    // pointer tp Database object
    Database* db;
    // pointer to Future object, to check if thread has been interrupted
    Future* future;
    // pointer to container object handle, to add processed rows data to the container
    rocprofvis_dm_handle_t handle;
    // callback method pointer
    RpvSqliteExecuteQueryCallback callback;
    // pointer to query string, convenient for multiuse callback debugging
    const char*              query[kRPVNumSourceQueryTypes];
    rocprofvis_dm_track_id_t track_id;
    rocprofvis_dm_event_operation_t operation;
} rocprofvis_db_sqlite_callback_parameters;

typedef std::vector<std::string> guid_list_t;
typedef std::map<void*, std::map<std::string, uint64_t>> rocprofvis_null_data_exceptions_int;
typedef std::map<void*, std::map<std::string, std::string>> rocprofvis_null_data_exceptions_string;
typedef std::map<void*, std::set<std::string>> rocprofvis_null_data_exceptions_skip;

// class for any Sqlite database methods and properties 
class SqliteDatabase : public Database
{
    public:
        // Database constructor
        // @param path - full path to database file
        SqliteDatabase( rocprofvis_db_filename_t path) : 
                        Database(path) {};
        // SqliteDatabase destructor, must be defined as virtual to free resources of derived classes 
        virtual ~SqliteDatabase() {Close();}
        // Method to open sqlite database
        // @return status of operation
        rocprofvis_dm_result_t Open() override;
        // Method to open sqlite database connection
        // @connection - pointer to connection
        // @return status of operation
        rocprofvis_dm_result_t OpenConnection(sqlite3** connection);
        // Method to close sqlite database
        // @return status of operation
        rocprofvis_dm_result_t Close() override;
        void  InterruptQuery(void* connection) override;

        void SetBlankMask(std::string op, uint64_t mask);
        void SetServiceMask(std::string op, uint64_t mask);
        void SetTimestampMask(std::string op, uint64_t mask);

        guid_list_t & GuidList() { return m_guid_list; } 

    protected:
        // Method to create SQL table
        // @param table_name - table name 
        // @param parameters - column insert parameters 
        // @param num_cols - number of columns
        // @param num_row - number of rows
        // @param insert_func - lambda method for data insertion
        // @return status of operation
        rocprofvis_dm_result_t CreateSQLTable(const char* table_name, 
                                              SQLInsertParams* parameters,
                                              uint8_t num_cols,
                                              size_t num_row,
                                              std::function<void(sqlite3_stmt* stmt, int index)> insert_func);
        // Method to delete SQL table
        // @param table_name - table name 
        // @return status of operation
        rocprofvis_dm_result_t DropSQLTable(const char* table_name);
        rocprofvis_dm_result_t DropSQLIndex(const char* table_name);
        // Method for SQL query execution without any callback
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query);
        // Method for SQL query execution with callback to process table rows 
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param callback - sqlite3_exec callback method for data processing
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                std::vector<std::string> query, 
                                                RpvSqliteExecuteQueryCallback callback);
        // Method for single row and column SQL query execution returning result of the query as string 
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param callback - sqlite3_exec callback method for data processing
        // @param value - pointer reference to string where the value will be placed
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query, 
                                                RpvSqliteExecuteQueryCallback callback,
                                                rocprofvis_dm_string_t* value);
        // Method for single row and column SQL query execution returning result of the query as uint64 
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param callback - sqlite3_exec callback method for data processing
        // @param value - pointer reference to uint64_t variable where the value will be placed
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query, 
                                                RpvSqliteExecuteQueryCallback callback,
                                                uint64_t & value);
        // Method for single row and column SQL query execution returning result of the query as uint32 
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param callback - sqlite3_exec callback method for data processing
        // @param value - pointer reference to uint32_t variable where the value will be placed
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query, 
                                                RpvSqliteExecuteQueryCallback callback,
                                                uint32_t & value);
        // Method for SQL query execution with  handle parameter. 
        // Used for callbacks storing data into container with rocprofvis_dm_handle_t handle
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param handle - handle of a container processed rows to be stored
        // @param callback - sqlite3_exec callback method for data processing
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query,
                                                rocprofvis_dm_handle_t handle, 
                                                RpvSqliteExecuteQueryCallback callback);
         // Method for SQL query execution with multi-use subquery parameter. 
        // Used for callbacks storing data into container with rocprofvis_dm_handle_t handle
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param subquery - multi-use parameter
        // @param callback - sqlite3_exec callback method for data processing
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query, 
                                                RpvSqliteExecuteQueryCallback callback);
        // Method for SQL query execution with multi-use subquery and handle parameter. 
        // Used for callbacks storing data into container with rocprofvis_dm_handle_t handle
        // @param future - future object for asynchronous execution status
        // @param query - SQL query
        // @param subquery - multi-use parameter
        // @param handle - handle of a container processed rows to be stored
        // @param callback - sqlite3_exec callback method for data processing
        // @return status of operation
        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, 
                                                const char* query,
                                                const char* cache_table_name,
                                                rocprofvis_dm_handle_t handle, 
                                                RpvSqliteExecuteQueryCallback callback);

        rocprofvis_dm_result_t ExecuteSQLQuery(Future* future, const char* query,
                                               const char*            cache_table_name,
                                               rocprofvis_dm_handle_t handle,
                                               rocprofvis_dm_event_operation_t op,
                                               RpvSqliteExecuteQueryCallback   callback);

        // Method to check if table exists in database
        // @param is_view - true if view
        // @param table - name of the table
        // @param conn - connection
        static int DetectTable(sqlite3* conn, const char* table, bool is_view = true);

        // sqlite3_exec callback to read value from single column and single row
        // @param data - pointer to callback caller argument
        // @param argc - number of columns in the query
        // @param argv - pointer to row values
        // @param azColName - pointer to column names  
        // @return SQLITE_OK if successful
        static int CallbackGetValue(void* data, int argc, sqlite3_stmt* stmt, char** azColName);  
        // sqlite3_exec callback to store all requested rows into Table container
        // @param data - pointer to callback caller argument
        // @param argc - number of columns in the query
        // @param argv - pointer to row values
        // @param azColName - pointer to column names  
        // @return SQLITE_OK if successful
        static int CallbackRunQuery(void *data, int argc, sqlite3_stmt* stmt, char **azColName); 
        // sqlite3_exec callback to write table query result to .CSV file with post processesing such that output matches tables in UI 
        // @param data - pointer to callback caller argument
        // @param argc - number of columns in the query
        // @param argv - pointer to row values
        // @param azColName - pointer to column names  
        // @return SQLITE_OK if successful
        static int CallbackTableQueryToCSV(void* data, int argc, sqlite3_stmt* stmt, char** azColName);
        // sqlite3_exec callback to write query result to .CSV file
        // @param data - pointer to callback caller argument
        // @param argc - number of columns in the query
        // @param argv - pointer to row values
        // @param azColName - pointer to column names  
        // @return SQLITE_OK if successful
        static int CallbackQueryToCSV(void* data, int argc, sqlite3_stmt* stmt, char** azColName);

        static rocprofvis_dm_result_t ExecuteSQLQueryStatic(
            SqliteDatabase* db, 
            Future* future, 
            const char* query,
            RpvSqliteExecuteQueryCallback callback);

        sqlite3* GetServiceConnection();

        rocprofvis_dm_result_t ExecuteTransaction(std::vector<std::string> queries);

        // method to run SQL query
        // @param db_conn - database connection 
        // @param query - SQL query
        // @param params - set of parameters to be passed to sqlite3_exec callback
        rocprofvis_dm_result_t ExecuteSQLQuery(const char* query, rocprofvis_db_sqlite_callback_parameters * params);

        static void CollectTrackServiceData(SqliteDatabase* db,
            sqlite3_stmt* stmt, int column_index, char** azColName,
            rocprofvis_db_sqlite_track_service_data_t& service_data);
        static void FindTrackIDs(
            SqliteDatabase* db, rocprofvis_db_sqlite_track_service_data_t& service_data,
            int& trackId, int & streamTrackId);
        rocprofvis_dm_track_category_t TranslateOperationToTrackCategory(rocprofvis_dm_event_operation_t op);
        static const rocprofvis_dm_track_search_id_t GetTrackSearchId(rocprofvis_dm_track_category_t category);
    
    protected:
        char* Sqlite3ColumnText(void* func, sqlite3_stmt* stmt, char** azColName, int index);
        int Sqlite3ColumnInt(void* func, sqlite3_stmt* stmt, char** azColName, int index);
        int64_t Sqlite3ColumnInt64(void* func, sqlite3_stmt* stmt, char** azColName, int index);
        double Sqlite3ColumnDouble(void* func, sqlite3_stmt* stmt, char** azColName, int index);
        uint64_t GetNullExceptionInt(void* func, char* column);
        char* GetNullExceptionString(void* func, char* column);
        bool NullExceptionSkip(void* func, char* column);
        virtual const rocprofvis_null_data_exceptions_int* GetNullDataExceptionsInt() = 0;
        virtual const rocprofvis_null_data_exceptions_string* GetNullDataExceptionsString() = 0;
        virtual const rocprofvis_null_data_exceptions_skip* GetNullDataExceptionsSkip() = 0;
        virtual const rocprofvis_dm_track_category_t GetRegionTrackCategory()    = 0;
    private:     

        std::set<sqlite3*> m_available_connections;
        std::set<sqlite3*> m_connections_inuse;
        std::mutex         m_mutex;
        std::condition_variable      m_inuse_cv;
        std::unordered_map<std::string, std::array<uint64_t, 3>> m_column_masks;
        guid_list_t        m_guid_list;

        // method to mimic slite3_exec using sqlite3_prepare_v2
        // @param db - database connection
        // @param query - SQL query
        // @param callback - sqlite3_exec type callback
        // @param user_data - user parameters
        int Sqlite3Exec(sqlite3* db, const char* query,
                                        int (*callback)(void*, int, sqlite3_stmt*, char**),
                                        void* user_data);
        sqlite3* GetConnection(); 
        
        void ReleaseConnection(sqlite3* conn);
        bool isServiceColumn(char* name);
        uint64_t GetBlanksMaskForQuery(std::string query);
        std::array<uint64_t, kRPVTableQueryColumnMaskCount> GetColumnMasksForQuery(std::string_view query);
};

}  // namespace DataModel
}  // namespace RocProfVis