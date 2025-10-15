# RDC Diagnostics API Tests - Data-Driven Architecture

Ultra-extensible, metadata-driven test framework for testing **all 194 RDC fields** without writing field-specific test code.

## 🎯 Design Philosophy

**Problem**: RDC has 194 telemetry fields. Writing individual tests for each field doesn't scale.

**Solution**: Data-driven parameterized tests with field metadata registry.

**Result**: Add new fields by updating metadata - zero new test code needed!

## 🏗️ Architecture

### Core Components

```
new_tests/
├── field_metadata.h/cc          # Field metadata system
│   ├── FieldMetadata             # Per-field metadata & validation
│   ├── FieldRegistry             # Central field database
│   └── FieldCategory             # Field categorization
├── rdc_test_fixture.h/cc        # Base test fixture (RAII)
├── test_field_telemetry.cc      # Parameterized field tests
└── test_field_categories.cc     # Category-based batch tests
```

### Adding New Fields

**Ultra-Easy - Just Add Metadata:**

```cpp
// In field_metadata.cc::InitializeFieldRegistry()
registry.Register(
    FieldMetadata(RDC_FI_MY_NEW_FIELD, "MY_NEW_FIELD", "Description",
                  FieldCategory::GPU_UTILIZATION, INTEGER)
        .WithRange(0, 100)
);
```

That's it! The field is automatically tested.

## 🚀 Running Tests

```bash
# Build
cmake -B build -DBUILD_TESTS=ON -DBUILD_STANDALONE=OFF -DBUILD_PROFILER=OFF
make -C build rdc_api_test

# Run all tests
./build/new_tests/rdc_api_test

# Run specific field
./build/new_tests/rdc_api_test --gtest_filter="*GPU_CLOCK*"

# Run category
./build/new_tests/rdc_api_test --gtest_filter="*GPU_FREQUENCY*"
```

See full documentation in README.md.old for complete details.
