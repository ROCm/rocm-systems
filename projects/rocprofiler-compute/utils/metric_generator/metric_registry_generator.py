"""
Hierarchical Metric Registry - Complete Solution
- Uses structured IDs: <panel_id>.<table_id>.<metric_index>
- Eliminates naming conflicts through positional identification
- Clean lookup tables with all metadata in main registry
- Only 2 files generated: registry + lookup tables
- Comprehensive logging with console limited to important messages
"""

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(root_dir: Path) -> logging.Logger:
    """Set up comprehensive logging with console limited to important messages"""
    log_dir = root_dir / "utils" / "metric_generator"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "registry_operations.log"

    # Create logger
    logger = logging.getLogger("hierarchical_registry")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler - captures everything
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - only important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


@dataclass
class HierarchicalMetricEntry:
    """Hierarchical metric entry with structured ID"""

    hierarchical_id: str  # e.g., "1300.1302.0"
    canonical_name: str  # e.g., "Req"
    panel_id: int
    table_id: int
    metric_index: int
    first_seen: str
    last_verified: str
    context: dict  # Panel/table context info

    def to_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "panel_id": self.panel_id,
            "table_id": self.table_id,
            "metric_index": self.metric_index,
            "first_seen": self.first_seen,
            "last_verified": self.last_verified,
            "context": self.context,
        }


class HierarchicalMetricRegistry:
    """Hierarchical metric registry with structured IDs"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.registry_file = (
            self.root_dir / "utils" / "metric_generator" / "metric_registry.yaml"
        )
        self.logger = setup_logging(root_dir)

        # Registry data - keyed by hierarchical ID
        self.metrics: dict[str, HierarchicalMetricEntry] = {}
        self.string_to_id_id: dict[str, list[str]] = {}
        self.registry_changed = False

        # Configuration
        self.today = datetime.date.today().isoformat()

        # Split YAML directories
        self.split_yaml_dir = (
            self.root_dir / "src" / "rocprof_compute_soc" / "analysis_configs"
        )
        self.architectures = [
            "gfx908",
            "gfx90a",
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx950",
        ]

        self.logger.info("Hierarchical Metric Registry initialized")
        self.logger.debug(f"Root directory: {self.root_dir}")
        self.logger.debug(f"Registry file: {self.registry_file}")

    def initialize_registry(self) -> bool:
        """Initialize registry - load existing or create from unified_config"""
        self.logger.info("Initializing hierarchical metric registry...")

        if self.registry_file.exists():
            self.logger.info("Loading existing registry")
            return self._load_existing_registry()
        else:
            self.logger.info("Creating initial registry from unified_config.yaml")
            return self._create_initial_registry()

    def _load_existing_registry(self) -> bool:
        """Load existing registry from file"""
        try:
            self.logger.debug(f"Reading registry file: {self.registry_file}")
            with open(self.registry_file) as f:
                data = yaml.safe_load(f)

            if not data or "metrics" not in data:
                self.logger.error("Registry file is empty or malformed")
                print("ERROR: Registry file is empty or malformed")
                print(
                    "REPAIR: Delete the registry file and "
                    "run with --force-create to regenerate"
                )
                return False

            # Load metrics
            for hierarchical_id, metric_data in data["metrics"].items():
                try:
                    # Parse hierarchical ID
                    parts = hierarchical_id.split(".")
                    if len(parts) != 3:
                        raise ValueError(
                            f"Invalid hierarchical ID format: {hierarchical_id}"
                        )

                    panel_id, table_id, metric_index = map(int, parts)

                    entry = HierarchicalMetricEntry(
                        hierarchical_id=hierarchical_id,
                        canonical_name=metric_data["canonical_name"],
                        panel_id=panel_id,
                        table_id=table_id,
                        metric_index=metric_index,
                        first_seen=metric_data["first_seen"],
                        last_verified=metric_data.get(
                            "last_verified", metric_data["first_seen"]
                        ),
                        context=metric_data.get("context", {}),
                    )

                    self.metrics[hierarchical_id] = entry

                    # Build reverse mapping (name -> list of hierarchical IDs)
                    if entry.canonical_name not in self.string_to_id_id:
                        self.string_to_id_id[entry.canonical_name] = []
                    self.string_to_id_id[entry.canonical_name].append(hierarchical_id)

                    self.logger.debug(
                        f"Loaded metric: {hierarchical_id} -> {entry.canonical_name}"
                    )

                except (ValueError, KeyError) as e:
                    self.logger.error(
                        f"Invalid registry entry for {hierarchical_id}: {e}"
                    )
                    print(f"ERROR: Invalid registry entry for {hierarchical_id}: {e}")
                    print(
                        "REPAIR: Fix the registry file manually or "
                        "regenerate with --force-create"
                    )
                    return False

            self.logger.info(f"Loaded registry with {len(self.metrics)} metrics")
            self._show_registry_summary()
            return True

        except yaml.YAMLError as e:
            self.logger.error(f"Cannot parse registry YAML: {e}")
            print(f"ERROR: Cannot parse registry YAML: {e}")
            print("REPAIR: Check YAML syntax or regenerate with --force-create")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
            print(f"ERROR: Failed to load registry: {e}")
            print("REPAIR: Check file permissions or regenerate with --force-create")
            return False

    def _create_initial_registry(self) -> bool:
        """Create initial registry from unified_config.yaml with hierarchical IDs"""
        unified_config_path = self.root_dir / "utils" / "unified_config.yaml"

        if not unified_config_path.exists():
            self.logger.error(f"unified_config.yaml not found at {unified_config_path}")
            print(f"ERROR: {unified_config_path} not found")
            print(
                "REPAIR: Ensure unified_config.yaml exists or "
                "run your existing generator first"
            )
            return False

        try:
            self.logger.info(f"Reading metrics from {unified_config_path}")

            # Read raw file content for line tracking
            with open(unified_config_path, "r") as f:
                raw_lines = f.readlines()

            # Parse YAML
            with open(unified_config_path) as f:
                config = yaml.safe_load(f)

            if not config or "panels" not in config:
                self.logger.error("unified_config.yaml is empty or has no panels")
                print("ERROR: unified_config.yaml is empty or has no panels")
                print("REPAIR: Generate valid unified_config.yaml first")
                return False

            # Extract metrics with hierarchical structure
            metrics_with_hierarchy = self._extract_metrics_with_hierarchy(
                config, raw_lines
            )

            if not metrics_with_hierarchy:
                self.logger.error("No metrics found in unified_config.yaml")
                print("ERROR: No metrics found in unified_config.yaml")
                print(
                    "REPAIR: Ensure unified_config.yaml contains valid "
                    "metric definitions"
                )
                return False

            self.logger.info(
                f"Found {len(metrics_with_hierarchy)} metrics with "
                "hierarchical structure"
            )

            # Register all metrics with hierarchical IDs
            for hierarchical_id, metric_info in metrics_with_hierarchy.items():
                self._register_hierarchical_metric(
                    hierarchical_id, metric_info, self.today
                )

            self.logger.info(
                f"Created initial registry with {len(metrics_with_hierarchy)} metrics"
            )
            self._show_registry_summary()
            self.registry_changed = True
            return True

        except yaml.YAMLError as e:
            self.logger.error(f"Cannot parse unified_config.yaml: {e}")
            print(f"ERROR: Cannot parse unified_config.yaml: {e}")
            print("REPAIR: Fix YAML syntax in unified_config.yaml")
            return False
        except Exception as e:
            self.logger.error(f"Failed to create initial registry: {e}")
            print(f"ERROR: Failed to create initial registry: {e}")
            print("REPAIR: Check file permissions and unified_config.yaml format")
            return False

    def _extract_metrics_with_hierarchy(
        self, config: dict, raw_lines: list[str]
    ) -> dict[str, dict]:
        """Extract metrics with hierarchical ID structure"""
        hierarchical_metrics = {}
        self.logger.debug("Extracting metrics with hierarchical structure")

        for panel in config.get("panels", []):
            panel_id = panel.get("id")
            panel_title = panel.get("title", "Unknown")

            if panel_id is None:
                self.logger.warning(f"Panel missing ID, skipping: {panel_title}")
                continue

            self.logger.debug(f"Processing panel {panel_id}: {panel_title}")

            for data_source in panel.get("data source", []):
                if "metric_table" in data_source:
                    metric_table = data_source["metric_table"]
                    table_id = metric_table.get("id")
                    table_title = metric_table.get("title", "Unknown")

                    if table_id is None:
                        self.logger.warning(
                            f"Metric table missing ID, skipping: {table_title}"
                        )
                        continue

                    self.logger.debug(f"Processing table {table_id}: {table_title}")

                    # Process each architecture to get metric names
                    metric_names_in_table = set()
                    architectures_for_table = []

                    for arch_name, arch_metrics in metric_table.get(
                        "metric", {}
                    ).items():
                        if isinstance(arch_metrics, dict):
                            architectures_for_table.append(arch_name)
                            for metric_name, metric_data in arch_metrics.items():
                                # Only process actual metric definitions
                                if self._is_valid_metric_definition(metric_data):
                                    metric_names_in_table.add(metric_name)
                                    self.logger.debug(
                                        f"Found metric: {metric_name} in {arch_name}"
                                    )

                    # Assign hierarchical IDs based on sorted metric names
                    sorted_metric_names = sorted(metric_names_in_table)
                    self.logger.debug(
                        f"Table {table_id} has {len(sorted_metric_names)} "
                        f"metrics: {sorted_metric_names}"
                    )

                    for metric_index, metric_name in enumerate(sorted_metric_names):
                        hierarchical_id = f"{panel_id}.{table_id}.{metric_index}"

                        # Find line numbers where this metric appears
                        line_numbers = self._find_metric_line_numbers(
                            metric_name, raw_lines
                        )

                        hierarchical_metrics[hierarchical_id] = {
                            "canonical_name": metric_name,
                            "panel_id": panel_id,
                            "table_id": table_id,
                            "metric_index": metric_index,
                            "architectures": sorted(architectures_for_table),
                            "context": {
                                "panel_title": panel_title,
                                "table_title": table_title,
                            },
                            "line_numbers": line_numbers,
                        }

                        self.logger.debug(
                            f"Created ID: {hierarchical_id} for {metric_name}"
                        )

        return hierarchical_metrics

    def _is_valid_metric_definition(self, metric_data: dict) -> bool:
        """Check if this is a valid metric definition and not just metadata"""
        if not isinstance(metric_data, dict):
            return False

        # Must have at least one of the metric value fields
        metric_value_fields = ["value", "avg", "min", "max", "formula"]
        has_metric_field = any(field in metric_data for field in metric_value_fields)

        # Should not be just metadata fields
        metadata_only_fields = ["unit", "peak", "pop", "header"]
        is_metadata_only = all(
            key in metadata_only_fields for key in metric_data.keys()
        )

        is_valid = has_metric_field and not is_metadata_only
        self.logger.debug(
            f"Metric validation: has_metric_field={has_metric_field}, "
            f"is_metadata_only={is_metadata_only}, valid={is_valid}"
        )
        return is_valid

    def _find_metric_line_numbers(
        self, metric_name: str, raw_lines: list[str]
    ) -> list[dict]:
        """Find all line numbers where a metric name appears with context"""
        occurrences = []

        for line_num, line in enumerate(raw_lines, 1):  # 1-based line numbers
            if (
                f"{metric_name}:" in line
                or f'"{metric_name}"' in line
                or f"'{metric_name}'" in line
            ):
                # Get surrounding context
                context_start = max(0, line_num - 3)
                context_end = min(len(raw_lines), line_num + 2)
                context_lines = []

                for ctx_line_num in range(context_start, context_end):
                    prefix = ">>>" if ctx_line_num == line_num - 1 else "   "
                    context_lines.append(
                        f"{prefix} {ctx_line_num + 1:4d}: "
                        f"{raw_lines[ctx_line_num].rstrip()}"
                    )

                occurrences.append({
                    "line_number": line_num,
                    "line_content": line.strip(),
                    "context": context_lines,
                })

                self.logger.debug(f"Found {metric_name} at line {line_num}")

        self.logger.debug(f"Metric {metric_name} found in {len(occurrences)} locations")
        return occurrences

    def _register_hierarchical_metric(
        self, hierarchical_id: str, metric_info: dict, date_seen: str
    ) -> None:
        """Register a metric with hierarchical ID"""
        entry = HierarchicalMetricEntry(
            hierarchical_id=hierarchical_id,
            canonical_name=metric_info["canonical_name"],
            panel_id=metric_info["panel_id"],
            table_id=metric_info["table_id"],
            metric_index=metric_info["metric_index"],
            first_seen=date_seen,
            last_verified=date_seen,
            context=metric_info["context"],
        )

        self.metrics[hierarchical_id] = entry
        self.registry_changed = True

        # Build reverse mapping
        if entry.canonical_name not in self.string_to_id_id:
            self.string_to_id_id[entry.canonical_name] = []
        self.string_to_id_id[entry.canonical_name].append(hierarchical_id)

        # Store source tracking for initial generation
        if not hasattr(self, "_source_tracking"):
            self._source_tracking = {}
        self._source_tracking[hierarchical_id] = metric_info

        # Log registration info
        self.logger.info(f"Registered: {hierarchical_id} -> {entry.canonical_name}")
        self.logger.debug(
            f"  Panel: {metric_info['context']['panel_title']} (ID: {entry.panel_id})"
        )
        self.logger.debug(
            f"  Table: {metric_info['context']['table_title']} (ID: {entry.table_id})"
        )
        self.logger.debug(f"  Position: {entry.metric_index} (in sorted order)")
        self.logger.debug(f"  Architectures: {', '.join(metric_info['architectures'])}")
        self.logger.debug(f"  Found in {len(metric_info['line_numbers'])} locations")

    def _show_registry_summary(self) -> None:
        """Show summary of registry structure"""
        # Group by panel
        panels = {}
        for entry in self.metrics.values():
            panel_id = entry.panel_id
            if panel_id not in panels:
                panels[panel_id] = {"tables": {}, "total_metrics": 0}

            table_id = entry.table_id
            if table_id not in panels[panel_id]["tables"]:
                panels[panel_id]["tables"][table_id] = 0

            panels[panel_id]["tables"][table_id] += 1
            panels[panel_id]["total_metrics"] += 1

        self.logger.info("Registry Structure:")
        for panel_id in sorted(panels.keys()):
            panel_info = panels[panel_id]
            summary = (
                f"Panel {panel_id}: {panel_info['total_metrics']} metrics "
                f"across {len(panel_info['tables'])} tables"
            )
            self.logger.info(f"  {summary}")
            for table_id in sorted(panel_info["tables"].keys()):
                metric_count = panel_info["tables"][table_id]
                self.logger.debug(f"    Table {table_id}: {metric_count} metrics")

    def get_metric_by_hierarchical_id(self, hierarchical_id: str) -> Optional[dict]:
        """Get metric information by hierarchical ID"""
        entry = self.metrics.get(hierarchical_id)
        if entry is None:
            self.logger.debug(f"Hierarchical ID not found: {hierarchical_id}")
            return None

        return {
            "hierarchical_id": entry.hierarchical_id,
            "canonical_name": entry.canonical_name,
            "panel_id": entry.panel_id,
            "table_id": entry.table_id,
            "metric_index": entry.metric_index,
            "context": entry.context,
            "first_seen": entry.first_seen,
            "last_verified": entry.last_verified,
        }

    def get_metrics_by_name(self, metric_name: str) -> list[dict]:
        """Get all metrics with the given name"""
        hierarchical_ids = self.string_to_id_id.get(metric_name, [])
        self.logger.debug(
            f"Found {len(hierarchical_ids)} metrics for "
            f"name '{metric_name}': {hierarchical_ids}"
        )
        return [self.get_metric_by_hierarchical_id(hid) for hid in hierarchical_ids]

    def validate_registry(self) -> bool:
        """Validate registry structure and consistency"""
        self.logger.info("Validating hierarchical metric registry...")

        validation_errors = []
        validation_warnings = []

        # 1. Validate hierarchical ID format
        self.logger.debug("Checking hierarchical ID formats...")
        for hierarchical_id, entry in self.metrics.items():
            parts = hierarchical_id.split(".")
            if len(parts) != 3:
                validation_errors.append(
                    f"Invalid hierarchical ID format: {hierarchical_id}"
                )
                continue

            try:
                panel_id, table_id, metric_index = map(int, parts)

                # Validate consistency with entry data
                if panel_id != entry.panel_id:
                    validation_errors.append(
                        f"Panel ID mismatch in {hierarchical_id}: "
                        f"ID says {panel_id}, entry says {entry.panel_id}"
                    )
                if table_id != entry.table_id:
                    validation_errors.append(
                        f"Table ID mismatch in {hierarchical_id}: "
                        f"ID says {table_id}, entry says {entry.table_id}"
                    )
                if metric_index != entry.metric_index:
                    validation_errors.append(
                        f"Metric index mismatch in {hierarchical_id}: "
                        f"ID says {metric_index}, entry says {entry.metric_index}"
                    )

            except ValueError:
                validation_errors.append(
                    f"Non-numeric components in hierarchical ID: {hierarchical_id}"
                )

        # 2. Check for duplicate hierarchical IDs
        self.logger.debug("Checking for duplicates...")
        id_counts = {}
        for hierarchical_id in self.metrics.keys():
            id_counts[hierarchical_id] = id_counts.get(hierarchical_id, 0) + 1

        duplicates = [hid for hid, count in id_counts.items() if count > 1]
        if duplicates:
            validation_errors.append(f"Duplicate hierarchical IDs found: {duplicates}")

        # 3. Validate metric indices are sequential within each table
        self.logger.debug("Checking metric index sequences...")
        table_metrics = {}
        for entry in self.metrics.values():
            table_key = f"{entry.panel_id}.{entry.table_id}"
            if table_key not in table_metrics:
                table_metrics[table_key] = []
            table_metrics[table_key].append(entry.metric_index)

        for table_key, indices in table_metrics.items():
            sorted_indices = sorted(indices)
            expected_indices = list(range(len(sorted_indices)))
            if sorted_indices != expected_indices:
                validation_warnings.append(
                    f"Non-sequential metric indices in "
                    f"table {table_key}: {sorted_indices}, expected {expected_indices}"
                )

        # 4. Check for name collisions within same context
        self.logger.debug("Checking for name collisions within contexts...")
        context_names = {}
        for entry in self.metrics.values():
            context_key = f"{entry.panel_id}.{entry.table_id}"
            if context_key not in context_names:
                context_names[context_key] = []
            context_names[context_key].append(entry.canonical_name)

        for context_key, names in context_names.items():
            name_counts = {}
            for name in names:
                name_counts[name] = name_counts.get(name, 0) + 1

            duplicates = [name for name, count in name_counts.items() if count > 1]
            if duplicates:
                validation_errors.append(
                    f"Duplicate metric names in context {context_key}: {duplicates}"
                )

        # Log all validation details
        for error in validation_errors:
            self.logger.error(f"Validation error: {error}")
        for warning in validation_warnings:
            self.logger.warning(f"Validation warning: {warning}")

        # Report results
        if validation_errors:
            self.logger.error(
                f"Registry validation FAILED with {len(validation_errors)} errors"
            )
            print("ERROR: REGISTRY VALIDATION FAILED")
            print(f"Found {len(validation_errors)} critical errors:")
            for i, error in enumerate(validation_errors, 1):
                print(f"  {i}. {error}")

            print("\nREPAIR INSTRUCTIONS:")
            print("  - Registry structure is corrupted")
            print("  - Recommended: Regenerate registry with --force-create")

            return False

        if validation_warnings:
            self.logger.warning(
                f"Registry validation passed with {len(validation_warnings)} warnings"
            )
            print(f"WARNING: Found {len(validation_warnings)} warnings:")
            for warning in validation_warnings[:3]:
                print(f"  - {warning}")
            if len(validation_warnings) > 3:
                print(f"  ... and {len(validation_warnings) - 3} more warnings")

        self.logger.info(
            f"Registry validation PASSED - {len(self.metrics)} metrics, "
            f"{len(self.string_to_id_id)} unique names"
        )
        print("INFO: Registry validation PASSED")

        # Show examples of name collisions (expected and handled)
        collisions = [
            name for name, ids in self.string_to_id_id.items() if len(ids) > 1
        ]
        if collisions:
            self.logger.info(
                f"Name collisions handled: {len(collisions)} names "
                "appear in multiple contexts"
            )
            for name in collisions:
                ids = self.string_to_id_id[name]
                self.logger.debug(f"  '{name}': {', '.join(ids)}")

        return True

    def save_registry(self) -> bool:
        """Save registry with hierarchical structure and comprehensive metadata"""
        if not self.registry_changed:
            self.logger.info("Registry unchanged, skipping save")
            return True

        try:
            self.logger.debug("Preparing registry data with comprehensive metadata")

            # Prepare registry data with comprehensive metadata
            registry_data = {
                "format_version": "1.0",
                "id_structure": "<panel_id>.<table_id>.<metric_index>",
                "generation_date": self.today,
                "total_metrics": len(self.metrics),
                "metrics": {},
            }

            # Add summary statistics if available
            if hasattr(self, "_source_tracking"):
                # Analyze structure for metadata
                panels = {}
                name_collisions = {}

                for hierarchical_id, source_info in self._source_tracking.items():
                    panel_id = source_info["panel_id"]
                    table_id = source_info["table_id"]
                    metric_name = source_info["canonical_name"]

                    # Panel stats
                    if panel_id not in panels:
                        panels[panel_id] = {
                            "title": source_info["context"]["panel_title"],
                            "tables": {},
                            "total_metrics": 0,
                        }
                    if table_id not in panels[panel_id]["tables"]:
                        panels[panel_id]["tables"][table_id] = {
                            "title": source_info["context"]["table_title"],
                            "metrics": 0,
                        }
                    panels[panel_id]["tables"][table_id]["metrics"] += 1
                    panels[panel_id]["total_metrics"] += 1

                    # Name collision tracking
                    if metric_name not in name_collisions:
                        name_collisions[metric_name] = []
                    name_collisions[metric_name].append(hierarchical_id)

                # Add metadata to registry
                registry_data["summary"] = {
                    "total_panels": len(panels),
                    "total_tables": sum(len(p["tables"]) for p in panels.values()),
                    "name_collisions_resolved": len([
                        name for name, ids in name_collisions.items() if len(ids) > 1
                    ]),
                }
                registry_data["structure"] = panels
                registry_data["name_collisions_resolved"] = {
                    name: ids for name, ids in name_collisions.items() if len(ids) > 1
                }

                self.logger.debug(
                    f"Added metadata: {len(panels)} panels, "
                    f"{registry_data['summary']['name_collisions_resolved']} "
                    "name collisions resolved"
                )

            # Sort metrics by hierarchical ID for consistent output
            for hierarchical_id in sorted(
                self.metrics.keys(), key=lambda x: [int(p) for p in x.split(".")]
            ):
                entry = self.metrics[hierarchical_id]
                metric_data = entry.to_dict()

                # Add source tracking if available
                if (
                    hasattr(self, "_source_tracking")
                    and hierarchical_id in self._source_tracking
                ):
                    source_info = self._source_tracking[hierarchical_id]
                    metric_data["source_tracking"] = {
                        "architectures": source_info["architectures"],
                        "line_occurrences": source_info["line_numbers"],
                    }

                registry_data["metrics"][hierarchical_id] = metric_data

            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)

            # Write registry file with all metadata and source tracking
            self.logger.debug(f"Writing registry file to {self.registry_file}")
            with open(self.registry_file, "w") as f:
                f.write("# Hierarchical Metric Registry - COMMIT TO GIT\n")
                f.write("# Uses structured IDs: <panel_id>.<table_id>.<metric_index>\n")
                f.write("# Contains ALL metadata and source tracking information\n")
                f.write("# DO NOT EDIT MANUALLY - Use registry tools\n")
                f.write(f"# Last updated: {self.today}\n")
                f.write(f"# Total metrics: {len(self.metrics)}\n\n")

                # Write source tracking information if this is initial generation
                if hasattr(self, "_source_tracking"):
                    f.write("# SOURCE TRACKING FOR INITIAL GENERATION\n")
                    f.write(
                        "# The following shows exactly where each hierarchical ID was "
                        "generated from\n\n"
                    )

                    for hierarchical_id in sorted(
                        self._source_tracking.keys(),
                        key=lambda x: [int(p) for p in x.split(".")],
                    ):
                        entry = self.metrics[hierarchical_id]
                        source_info = self._source_tracking[hierarchical_id]

                        f.write(
                            f"# Metric ID {hierarchical_id}: {entry.canonical_name}\n"
                        )
                        f.write(
                            f"# Panel: {source_info['context']['panel_title']} "
                            f"(ID: {entry.panel_id})\n"
                        )
                        f.write(
                            f"# Table: {source_info['context']['table_title']} "
                            f"(ID: {entry.table_id})\n"
                        )
                        f.write(
                            "# Architectures: "
                            f"{', '.join(source_info['architectures'])}\n"
                        )
                        f.write("# Found in unified_config.yaml at:\n")

                        for occurrence in source_info["line_numbers"]:
                            f.write(
                                f"#   Line {occurrence['line_number']}: "
                                f"{occurrence['line_content']}\n"
                            )
                            f.write("#   Context:\n")
                            for context_line in occurrence["context"]:
                                f.write(f"#   {context_line}\n")
                            f.write("#\n")

                        f.write("#\n")

                    f.write("# END SOURCE TRACKING\n\n")

                yaml.dump(registry_data, f, sort_keys=False, default_flow_style=False)

            self.logger.info(f"Registry saved to {self.registry_file}")
            self.logger.info(
                f"Registry contains {len(self.metrics)} metrics with "
                f"hierarchical structure"
            )
            print(f"INFO: Registry saved: {len(self.metrics)} metrics")

            self.registry_changed = False
            return True

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
            print(f"ERROR: Failed to save registry: {e}")
            print("REPAIR: Check file permissions and disk space")
            return False

    def generate_lookup_tables(self) -> bool:
        """Generate clean, minimal lookup tables without metadata"""
        lookup_file = (
            self.root_dir / "utils" / "metric_generator" / "metric_lookup_tables.yaml"
        )

        try:
            self.logger.debug("Generating clean lookup tables")

            # Build lookup tables
            id_to_string = {}
            string_to_id = {}

            # Sort hierarchical IDs for consistent output
            for hierarchical_id in sorted(
                self.metrics.keys(), key=lambda x: [int(p) for p in x.split(".")]
            ):
                entry = self.metrics[hierarchical_id]

                # Simple bidirectional mapping
                id_to_string[hierarchical_id] = entry.canonical_name

                # For name to hierarchical, we can have multiple contexts
                if entry.canonical_name not in string_to_id:
                    string_to_id[entry.canonical_name] = []
                string_to_id[entry.canonical_name].append(hierarchical_id)

            # Clean lookup data - just the mappings
            lookup_data = {
                "id_to_string": id_to_string,
                "string_to_id": string_to_id,
            }

            lookup_file.parent.mkdir(parents=True, exist_ok=True)

            self.logger.debug(f"Writing lookup tables to {lookup_file}")
            with open(lookup_file, "w") as f:
                f.write("# Metric Lookup Tables (bidirectional)\n")
                f.write("# Format: <panel_id>.<table_id>.<metric_index>\n\n")
                yaml.dump(
                    lookup_data, f, sort_keys=False, default_flow_style=False, width=120
                )

            self.logger.info(f"Clean lookup tables saved to {lookup_file}")
            print("INFO: Lookup tables saved")
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate lookup tables: {e}")
            print(f"ERROR: Failed to generate lookup tables: {e}")
            return False


def main():
    """Main hierarchical registry operations"""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical Metric Registry")
    parser.add_argument(
        "--root-dir", type=Path, default=Path.cwd(), help="Project root directory"
    )

    # Operations
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize registry (load existing or create)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate registry")
    parser.add_argument(
        "--force-create",
        action="store_true",
        help="Force create new registry (deletes existing)",
    )
    parser.add_argument("--info", help="Get info for metric (hierarchical ID or name)")
    parser.add_argument(
        "--structure", action="store_true", help="Show registry structure"
    )

    args = parser.parse_args()

    registry = HierarchicalMetricRegistry(args.root_dir)

    # Force create - delete existing registry
    if args.force_create:
        if registry.registry_file.exists():
            registry.registry_file.unlink()
            registry.logger.info(f"Deleted existing registry: {registry.registry_file}")
            print("INFO: Deleted existing registry")

        registry.logger.info(
            "Creating hierarchical registry with detailed source tracking..."
        )
        if not registry.initialize_registry():
            print("ERROR: Failed to create registry")
            return 1

        if not registry.save_registry():
            return 1

        if not registry.generate_lookup_tables():
            return 1

        print("INFO: HIERARCHICAL REGISTRY CREATED SUCCESSFULLY")
        print(f"INFO: Total metrics: {len(registry.metrics)}")
        print("INFO: ID structure: <panel_id>.<table_id>.<metric_index>")

        # Show sample of hierarchical ID assignments
        sample_count = min(5, len(registry.metrics))
        print("\nINFO: Sample hierarchical ID assignments:")
        for hierarchical_id in sorted(
            list(registry.metrics.keys()), key=lambda x: [int(p) for p in x.split(".")]
        )[:sample_count]:
            entry = registry.metrics[hierarchical_id]
            print(f"  {hierarchical_id}: {entry.canonical_name}")

        if len(registry.metrics) > sample_count:
            print(f"  ... and {len(registry.metrics) - sample_count} more metrics")

        # Show name collision examples
        collisions = [
            name for name, ids in registry.string_to_id_id.items() if len(ids) > 1
        ]
        if collisions:
            print("\nINFO: Name collisions resolved by hierarchical structure:")
            for name in collisions[:3]:
                ids = registry.string_to_id_id[name]
                print(f"  '{name}': {', '.join(ids)}")
            if len(collisions) > 3:
                print(f"  ... and {len(collisions) - 3} more resolved")
        else:
            print("\nINFO: Zero naming conflicts detected!")

        print("\nINFO: IMPORTANT - Commit files to git:")
        print("   git add utils/metric_generator/metric_registry.yaml")
        print("   git add utils/metric_generator/metric_lookup_tables.yaml")
        print("   git commit -m 'Add hierarchical metric registry'")
        print("\nINFO: Log file: utils/metric_generator/registry_operations.log")

        return 0

    # Initialize registry
    if args.init or args.validate or args.info or args.structure:
        if not registry.initialize_registry():
            print("ERROR: Failed to initialize registry")
            return 1

    # Validate registry
    if args.validate:
        if not registry.validate_registry():
            print("ERROR: Registry validation failed")
            return 1

    # Show structure
    if args.structure:
        registry._show_registry_summary()

        # Show collision examples
        collisions = [
            name for name, ids in registry.string_to_id_id.items() if len(ids) > 1
        ]
        if collisions:
            print("\nINFO: Name collisions handled by hierarchical IDs:")
            for name in collisions[:5]:
                ids = registry.string_to_id_id[name]
                print(f"  '{name}': {', '.join(ids)}")
                for hid in ids:
                    entry = registry.metrics[hid]
                    print(
                        f"    {hid}: {entry.context.get('panel_title', 'Unknown')} -> "
                        f"{entry.context.get('table_title', 'Unknown')}"
                    )
            if len(collisions) > 5:
                print(f"  ... and {len(collisions) - 5} more")

    # Get metric info
    if args.info:
        try:
            # Check if it's a hierarchical ID (contains dots)
            if "." in args.info:
                # Hierarchical ID lookup
                info = registry.get_metric_by_hierarchical_id(args.info)
                if info:
                    print(f"INFO: Metric Information (Hierarchical ID: {args.info}):")
                    print(f"  Name: {info['canonical_name']}")
                    print(
                        f"  Panel: {info['context'].get('panel_title', 'Unknown')} "
                        f"(ID: {info['panel_id']})"
                    )
                    print(
                        f"  Table: {info['context'].get('table_title', 'Unknown')} "
                        f"(ID: {info['table_id']})"
                    )
                    print(f"  Position: {info['metric_index']}")
                else:
                    print(f"ERROR: Hierarchical ID not found: {args.info}")
                    return 1
            else:
                # Name lookup
                infos = registry.get_metrics_by_name(args.info)
                if infos:
                    if len(infos) == 1:
                        info = infos[0]
                        print(f"INFO: Metric Information (Name: '{args.info}'):")
                        print(f"  Hierarchical ID: {info['hierarchical_id']}")
                        print(
                            f"  Panel: {info['context'].get('panel_title', 'Unknown')} "
                            f"(ID: {info['panel_id']})"
                        )
                        print(
                            f"  Table: {info['context'].get('table_title', 'Unknown')} "
                            f"(ID: {info['table_id']})"
                        )
                        print(f"  Position: {info['metric_index']}")
                    else:
                        print(f"INFO: Multiple metrics found for name '{args.info}':")
                        for info in infos:
                            print(
                                f"  {info['hierarchical_id']}: "
                                f"{info['context'].get('panel_title', 'Unknown')} -> "
                                f"{info['context'].get('table_title', 'Unknown')}"
                            )
                else:
                    print(f"ERROR: Metric name not found: {args.info}")
                    return 1

        except Exception as e:
            registry.logger.error(f"Error getting metric info: {e}")
            print(f"ERROR: {e}")
            return 1

    # Default action - init + validate
    if not any([args.force_create, args.validate, args.info, args.structure]):
        if not registry.initialize_registry():
            return 1

        if not registry.validate_registry():
            return 1

        print("INFO: Registry ready")

    print("INFO: Detailed logs: utils/metric_generator/registry_operations.log")
    return 0


if __name__ == "__main__":
    exit(main())
