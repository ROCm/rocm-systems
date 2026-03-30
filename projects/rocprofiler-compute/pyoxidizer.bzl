# PyOxidizer configuration for rocprof-compute
# This configuration builds a standalone binary for the rocprof-compute application

def make_exe():
    # Use default Python distribution (3.10)
    dist = default_python_distribution()

    policy = dist.make_python_packaging_policy()
    
    # Allow loading extension modules from memory
    policy.extension_module_filter = "all"
    policy.resources_location_fallback = "filesystem-relative:lib"
    
    # Include all package resources
    policy.include_distribution_resources = True
    policy.include_distribution_sources = False
    policy.include_test = False
    
    python_config = dist.make_python_interpreter_config()
    
    # Optimize for performance
    python_config.optimization_level = 2
    
    # Allow filesystem access for configuration files and data
    python_config.filesystem_importer = True
    
    # Disable isolated mode to allow proper encoding initialization
    python_config.isolated = False
    
    # Use environment to avoid conflicts with system Python
    python_config.use_environment = False
    
    # Entry module (replaces former src/__main__.py)
    python_config.run_module = "rocprof_compute_main"
    
    exe = dist.to_python_executable(
        name="rocprof-compute",
        packaging_policy=policy,
        config=python_config,
    )
    
    # Add all required Python packages from requirements.txt
    for resource in exe.pip_install([
        "astunparse==1.6.2",
        "colorlover",
        "dash-bootstrap-components",
        "dash-svg",
        "dash>=3.0.0",
        "kaleido==0.2.1",
        "matplotlib",
        "numpy>=1.17.5",
        "pandas>=1.4.3",
        "plotext",
        "plotille",
        "pymongo",
        "pyyaml",
        "setuptools",
        "sqlalchemy>=2.0.42",
        "tabulate",
        "textual",
        "textual_plotext",
        "textual-fspicker>=0.4.3",
        "tqdm",
    ]):
        exe.add_python_resource(resource)
    
    # Add project source files and packages - embed in memory for true standalone binary
    for resource in exe.read_package_root(
        path="src",
        packages=[
            "rocprof_compute_analyze",
            "rocprof_compute_profile",
            "rocprof_compute_tui",
            "rocprof_compute_soc",
            "utils",
        ],
    ):
        # Force all resources to be in-memory for fully self-contained binary
        resource.add_location = "in-memory"
        exe.add_python_resource(resource)
    
    # Read all Python files from src directory
    for resource in exe.read_package_root(
        path="src",
        packages=[],
    ):
        resource.add_location = "in-memory"
        exe.add_python_resource(resource)
    
    return exe

def make_embedded_resources(exe):
    return exe.to_embedded_resources()

def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files

register_target("exe", make_exe)
register_target("resources", make_embedded_resources, depends=["exe"], default_build_script=True)
register_target("install", make_install, depends=["exe"], default=True)

resolve_targets()
