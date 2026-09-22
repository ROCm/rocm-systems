# RDC REST API

## Overview
This REST API provides functionalities to:
- Discover available GPUs on a node.
- Configure and manage GPU monitoring queries.
- Retrieve GPU metrics based on configured queries.

The API is built using Flask and interacts with the RDC library to monitor GPU usage and performance metrics.

## Installation
### Prerequisites
- Python 3.x
- Flask
- RDC Library (`librdc_bootstrap.so` must be available and accessible)

### Install Dependencies
```sh
pip install flask
```

## Running the API
1. Ensure `librdc_bootstrap.so` is in the library path:
   ```sh
   export LD_LIBRARY_PATH=/path/to/librdc_bootstrap.so:$LD_LIBRARY_PATH
   ```
2. Run the API:
   ```sh
   python rdc_rest_api.py
   ```

The API will start and listen on `http://0.0.0.0:50052`.

## API Endpoints

### 1. Discover GPUs
**GET** `/rdc/discovery`
#### Response:
```json
{
  "0": "GPU Name",
  "1": "GPU Name"
}
```

### 2. Create Query Criteria
**POST** `/rdc/query_criteria`
#### Request Body:
```json
{
  "gpu_index": [0,1],
  "metrics": ["RDC_FI_GPU_CLOCK", "RDC_FI_GPU_TEMP"]
}
```
#### Response:
```json
{
  "query_id": "G-1-F-2"
}
```

### 3. Get Query Criteria
**GET** `/rdc/query_criteria/<query_id>`
#### Response:
```json
{
  "gpu_index": [0,1],
  "metrics": ["RDC_FI_GPU_CLOCK", "RDC_FI_GPU_TEMP"],
  "query_id": "G-1-F-2"
}
```

### 4. Delete Query Criteria
**DELETE** `/rdc/query_criteria/<query_id>`
#### Response:
```json
{
  "message": "Deleted successfully"
}
```

### 5. Retrieve GPU Metrics
**GET** `/rdc/gpu_metrics/<query_id>`
#### Response:
```json
[
  {
    "gpu_index": 0,
    "RDC_FI_GPU_CLOCK": 1450,
    "RDC_FI_GPU_TEMP": 32
  },
  {
    "gpu_index": 1,
    "RDC_FI_GPU_CLOCK": 736,
    "RDC_FI_GPU_TEMP": 35
  }
]
```

## Notes
- Ensure `librdc_bootstrap.so` is properly linked.
- The API should be run on a system with RDC installed and GPUs accessible.
