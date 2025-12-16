# ROCpd Validation Flow

```mermaid
flowchart TD
    A[Start: validate-rocpd.py] --> B{Parse Arguments}
    B --> |--help| C[Display Help & Exit]
    B --> |Missing --database| D[Show Error & Exit]
    B --> |Valid Args| E[Load Validation Rules]

    E --> F{Rules File Exists?}
    F --> |No| G[Use Default Rules<br/>default_rules.json]
    F --> |Yes| H[Load Custom Rules]
    G --> I[Parse JSON Rules]
    H --> I

    I --> J[Create Rule Objects:<br/>• required_table<br/>• validation_rule]

    J --> K{Database File Exists?}
    K --> |No| L[Error: File Not Found]
    K --> |Yes| M[Connect to SQLite Database]

    M --> N[Get All Tables from Database<br/>SELECT name FROM sqlite_master]

    N --> O[Start Validation Loop]
    O --> P[For Each Required Table Rule]

    P --> Q{Table Exists<br/>in Database?}
    Q --> |No| R[❌ FAIL: Table Missing]
    Q --> |Yes| S[Check Required Columns<br/>PRAGMA table_info]

    S --> T{All Required<br/>Columns Present?}
    T --> |No| U[❌ FAIL: Missing Columns]
    T --> |Yes| V[Check Minimum Row Count]

    V --> W{Meets Minimum<br/>Row Count?}
    W --> |No| X[❌ FAIL: Insufficient Rows]
    W --> |Yes| Y[Execute Validation Queries]

    Y --> Z[For Each Query in Rule]
    Z --> AA[Execute SQL Query]
    AA --> BB[Get Result]
    BB --> CC{Validation<br/>Comparison Pass?}

    CC --> |No| DD[❌ FAIL: Query Failed<br/>Log Error Message]
    CC --> |Yes| EE[✅ PASS: Query Passed]

    EE --> FF{More Queries?}
    DD --> FF
    FF --> |Yes| Z
    FF --> |No| GG{More Tables?}

    R --> GG
    U --> GG
    X --> GG

    GG --> |Yes| P
    GG --> |No| HH{All Validations<br/>Passed?}

    HH --> |Yes| II[✅ SUCCESS<br/>Exit Code: 0]
    HH --> |No| JJ[❌ FAILURE<br/>Exit Code: 65]

    L --> KK[Exit Code: 1]

    subgraph "Validation Rules Structure"
        LL[JSON Rules File]
        LL --> MM["required_tables[]"]
        MM --> NN["Table Definition:<br/>• name<br/>• required_columns<br/>• min_rows<br/>• validation_queries"]
        NN --> OO["Validation Query:<br/>• description<br/>• query (SQL)<br/>• expected_result<br/>• comparison<br/>• error_message"]
    end

    subgraph "Database Structure"
        PP[ROCpd SQLite Database]
        PP --> QQ[Tables:<br/>• kernel_summary<br/>• kernels<br/>• threads<br/>• ...]
        QQ --> RR[Columns per Table]
        RR --> SS[Data Rows]
    end

    subgraph "Comparison Operations"
        TT[Supported Comparisons:<br/>• equals<br/>• greater_than<br/>• less_than<br/>• greater_than_or_equal<br/>• less_than_or_equal<br/>• not_equals]
    end
```

## Input Phase

- Takes a ROCpd database file (.db) as input
- Optionally accepts custom validation rules (JSON file)
- Uses default rules if no custom rules provided

## Validation Rules Structure

- JSON-based configuration with required tables
- Each table has:
  - Required columns to check for
  - Minimum row count requirements
  - Custom SQL validation queries

## Validation Process

- For each required table, the tool:

  - Checks table existence in the database
  - Verifies required columns are present
  - Validates minimum row count
  - Executes custom SQL queries with various comparison operations

## Output & Results

- Real-time feedback with ✅/❌ indicators
- Detailed error messages for failures
- Exit codes:
  - **0**: All validations passed
  - **65**: Validation failures
  - **1**: General errors (file not found, etc.)
