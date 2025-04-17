#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from flask import Flask, request, jsonify
from RdcReader import RdcReader
from RdcUtil import RdcUtil
from rdc_bootstrap import *

# Initialize Flask app
app = Flask(__name__)

# Initialize RDC Reader and Utilities for handling GPU queries
rdc_reader = RdcReader(ip_port=None)
rdc_util = RdcUtil()

# Dictionary to store query criteria with query_id
gpu_queries = {}

# Endpoint to discover available GPUs
@app.route('/rdc/discovery', methods=['GET'])
def discover_gpus():
    """Retrieve a list of available GPUs and their names."""
    try:
        gpu_indexes = rdc_util.get_all_gpu_indexes(rdc_reader.rdc_handle)
        gpus = {}
        for gpu in gpu_indexes:
            device_attr = rdc_device_attributes_t()
            rdc.rdc_device_get_attributes(rdc_reader.rdc_handle, gpu, device_attr)
            gpus[gpu] = device_attr.device_name.decode('utf-8')  # Decode GPU name from bytes
        return jsonify(gpus)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to create a new query criteria
@app.route('/rdc/query_criteria', methods=['POST'])
def create_query_criteria():
    """Define a new query criteria specifying GPU indices and metrics to monitor."""
    try:
        data = request.json
        if not data or "metrics" not in data:
            return jsonify({"error": "Invalid request payload"}), 400
        
        gpu_indexes = data.get("gpu_index", rdc_util.get_all_gpu_indexes(rdc_reader.rdc_handle))
        metrics = data.get("metrics", [])
        
        # Create rdc group and fieldgroup
        gpu_group_id, _ = rdc_util.create_gpu_group(rdc_reader.rdc_handle, b"query_gpu_group", gpu_indexes)
        field_group_id, _ = rdc_util.create_field_group(rdc_reader.rdc_handle, b"query_field_group", [rdc.get_field_id_from_name(m.encode('utf-8')).value for m in metrics])

        # Call rdc_field_watch to start fetching metrics into cache
        result = rdc.rdc_field_watch(rdc_reader.rdc_handle, gpu_group_id, field_group_id, 1000000, 3600.0, 1000)
        if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
            return jsonify({"error": "Failed to watch fields"}), 500
        
        query_id = f"G-{gpu_group_id.value}-F-{field_group_id.value}"
        gpu_queries[query_id] = {"gpu_index": gpu_indexes, "metrics": metrics, "query_id": query_id}
        return jsonify({"query_id": query_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to get all query criteria
@app.route('/rdc/query_criteria', methods=['GET'])
def get_all_query_criteria():
    """Retrieve all stored query criteria for all GPUs."""
    try:
        query_id = request.args.get("query_id")
        if query_id:
            return jsonify(gpu_queries.get(query_id, {}))
        return jsonify(list(gpu_queries.values()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to retrieve a specific query criteria
@app.route('/rdc/query_criteria/<query_id>', methods=['GET'])
def get_query_criteria(query_id):
    """Retrieve query criteria based on a given query ID."""
    try:
        if query_id in gpu_queries:
            return jsonify(gpu_queries[query_id])
        return jsonify({"error": "Query ID not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to delete a specific query criteria
@app.route('/rdc/query_criteria/<query_id>', methods=['DELETE'])
def delete_query_criteria(query_id):
    """Delete a query criteria using its query ID."""
    try:
        if query_id in gpu_queries:
            gpu_group_id = rdc_reader.field_group_id
            field_group_id = rdc_reader.field_group_id
            
            # Call rdc_field_unwatch to stop fetching metrics
            result = rdc.rdc_field_unwatch(rdc_reader.rdc_handle, gpu_group_id, field_group_id)
            if rdc_status_t(result) != rdc_status_t.RDC_ST_OK:
                return jsonify({"error": "Failed to unwatch fields"}), 500
            
            # Delete GPU and field groups
            rdc.rdc_group_gpu_destroy(rdc_reader.rdc_handle, gpu_group_id)
            rdc.rdc_group_field_destroy(rdc_reader.rdc_handle, field_group_id)
            
            # Remove the query from storage
            del gpu_queries[query_id]
            return jsonify({"message": "Deleted successfully"})
        return jsonify({"error": "Query ID not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to fetch GPU metrics for a specific query ID
@app.route('/rdc/gpu_metrics/<query_id>', methods=['GET'])
def get_gpu_metrics(query_id):
    """Retrieve GPU metrics based on the query ID."""
    try:
        if query_id not in gpu_queries:
            return jsonify({"error": "Query ID not found"}), 404
        
        query = gpu_queries[query_id]
        gpu_metrics = []  # List to store GPU metric results
        for gpu in query["gpu_index"]:
            gpu_data = {"gpu_index": gpu}  # Store GPU index in the response
            for metric in query["metrics"]:
                field_id = rdc.get_field_id_from_name(metric.encode('utf-8')).value
                value = rdc_field_value()
                result = rdc.rdc_field_get_latest_value(rdc_reader.rdc_handle, gpu, field_id, value)
                if rdc_status_t(result) == rdc_status_t.RDC_ST_OK:
                    gpu_data[metric] = value.value.l_int  # Store metric value
            gpu_metrics.append(gpu_data)  # Append GPU data to results
        return jsonify(gpu_metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point to start the Flask server
if __name__ == '__main__':
    # Runs the API server, making it accessible on all network interfaces
    app.run(host='0.0.0.0', port=50052)
