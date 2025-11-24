#!/usr/bin/env bash

set -e

# Base URL for ROCm version repo
BASE_URL="https://repo.radeon.com/amdgpu-install"

# Command line arguments
DISTRO=""
OS_VERSION=""
ROCM_VERSION=""

# Default bounds for ROCm versions
LOWER_BOUND="5.3"
UPPER_BOUND=""
ROCPROFILER_SYSTEMS_LOWER_BOUND="6.3"
UPDATE_CONTAINERS=false
OMIT_PATCH_VERSIONS=true  # By default, omit patch versions (e.g., 6.3.1)
OUTPUT_FORMAT="matrix"  # or "list", "yaml"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

usage() {
    print_option() { printf "    --%-20s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    print_default_option() { printf "    --%-20s %-24s     %s (default: %s)\n" "${1}" "${2}" "${3}" "${4}"; }
    
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Query supported ROCm versions and OS combinations from AMD repository."
    echo ""
    echo "Options:"
    print_option "help -h" "" "This message"
    print_option "update-containers" "" "Generate containers.yml with supported combinations"
    print_option "include-patch-versions" "" "Include patch versions (e.g., 6.3.1) in output"
    
    echo ""
    print_option "distro" "[ubuntu|rhel|opensuse|el]" "OS distribution"
    print_option "version" "VERSION" "OS version (e.g., 22.04, jammy, 9.4, 15.6)"
    print_option "rocm-version" "VERSION" "ROCm version to query (e.g., 6.3, 7.0, 7.1)"
    print_default_option "lower-bound -l" "VERSION" "Minimum ROCm version to consider" "${LOWER_BOUND}"
    print_default_option "upper-bound -u" "VERSION" "Maximum ROCm version to consider" "${UPPER_BOUND:-latest}"
    print_default_option "format" "[matrix|list|yaml]" "Output format" "${OUTPUT_FORMAT}"
    
    echo ""
    echo "Ubuntu Codenames:"
    echo "    20.04 = focal    22.04 = jammy    24.04 = noble"
    
    echo ""
    echo "Examples:"
    echo "    # List all ROCm versions supported on Ubuntu 22.04 (or jammy)"
    echo "    $0 --distro ubuntu --version 22.04"
    echo "    $0 --distro ubuntu --version jammy"
    echo ""
    echo "    # List all OS distributions and versions that support ROCm 7.0"
    echo "    $0 --rocm-version 7.0"
    echo ""
    echo "    # Generate containers.yml (omits patch versions by default)"
    echo "    $0 --update-containers > containers.yml"
    echo ""
    echo "    # Generate containers.yml with patch versions included"
    echo "    $0 --update-containers --include-patch-versions > containers.yml"
    echo ""
    echo "    # Get combinations in YAML format with custom bounds"
    echo "    $0 --rocm-version 6.4 --format yaml"
    
    exit 0
}

# Ubuntu helper functions, because they use the codenames instead of versions
ubuntu_version_to_codename() {
    case "$1" in
        20.04) echo "focal" ;;
        22.04) echo "jammy" ;;
        24.04) echo "noble" ;;  
        focal|jammy|noble) echo "$1" ;;
        *) echo "Unknown/Unsupported Ubuntu version $1" ;;
    esac
}

ubuntu_codename_to_version() {
        case "$1" in
        focal) echo "20.04" ;;
        jammy) echo "22.04" ;;
        noble) echo "24.04" ;;
        *) echo "Unknown/Unsupported Ubuntu codename $1" ;;
    esac
}

# Convert OS version to repo format (Ubuntu)
os_version_to_repo() {
    local distro="$1"
    local version="$2"

    if [[ "$distro" == "ubuntu" ]]; then
        ubuntu_version_to_codename "$version"
    else
        echo "$version"
    fi
}

# Display version in friendly format (Ubuntu)
os_version_from_repo() {
    local distro="$1"
    local version="$2"

    if [[ "$distro" == "ubuntu" ]]; then
        ubuntu_codename_to_version "$version"
    else
        echo "$version"
    fi
}

tolower()
{
    echo "$@" | awk -F '\\|~\\|' '{print tolower($1)}';
}

toupper()
{
    echo "$@" | awk -F '\\|~\\|' '{print toupper($1)}';
}

# We do not error out here, that way if any other distros are added, the script can still run
distro_to_repo() {
    case "$1" in
        ubuntu) echo "ubuntu" ;;
        rhel) echo "rhel" ;;
        opensuse) echo "sle" ;;
        sle) echo "sle" ;;
        el) echo "el" ;;
        *) echo "$1" ;;
    esac
}

# Map repo directory names to friendly distro names
repo_to_distro() {
    case "$1" in
        ubuntu) echo "ubuntu" ;;
        rhel) echo "rhel" ;;
        sle) echo "opensuse" ;;
        el) echo "el" ;;
        *) echo "$1" ;;
    esac
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --distro)
                DISTRO="$2"
                shift 2
                ;;
            --version)
                OS_VERSION="$2"
                shift 2
                ;;
            --rocm-version)
                ROCM_VERSION="$2"
                shift 2
                ;;
            --lower-bound|-l)
                LOWER_BOUND="$2"
                shift 2
                ;;
            --upper-bound|-u)
                UPPER_BOUND="$2"
                shift 2
                ;;
            --format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --update-containers)
                UPDATE_CONTAINERS=true
                shift 1
                ;;
            --include-patch-versions)
                OMIT_PATCH_VERSIONS=false
                shift 1
                ;;
            --help|-h)
                usage
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}" >&2
                usage
                ;;
        esac
    done
}

get_rocm_versions() {
    curl -s "$BASE_URL/" | \
        grep -oP '<a href="\K[0-9]+\.[0-9]+(\.[0-9]+)?(?=/")' | \
        sort -V | \
        uniq | \
        # Filter out AMDGPU installer versions (they start with 21.x and above)
        # ROCm versions are in the 5.x through latest
        # We keep versions starting with 5-9 or 10-19
        grep -E '^([5-9]|1[0-9])\.'
}

# Compare versions: returns 0 if v1 >= v2
version_gte() {
    local v1="$1"
    local v2="$2"

    if [[ "$(printf '%s\n' "$v1" "$v2" | sort -V | head -n1)" == "$v2" ]]; then
        return 0
    else
        return 1
    fi
}

# Compare versions: returns 0 if v1 <= v2
version_lte() {
    local v1="$1"
    local v2="$2"

    if [[ "$(printf '%s\n' "$v1" "$v2" | sort -V | tail -n1)" == "$v2" ]]; then
        return 0
    else
        return 1
    fi
}

# Filter versions based on bounds
filter_versions() {
    local versions="$1"
    
    while IFS= read -r version; do
        if [[ -n "$LOWER_BOUND" ]]; then
            if ! version_gte "$version" "$LOWER_BOUND"; then
                continue
            fi
        fi
        
        if [[ -n "$UPPER_BOUND" ]]; then
            if ! version_lte "$version" "$UPPER_BOUND"; then
                continue
            fi
        fi
        
        echo "$version"
    done <<< "$versions"
}

# Get OS types available for a specific ROCm version
get_os_types_for_rocm() {
    local rocm_version="$1"
    curl -s "$BASE_URL/$rocm_version/" | \
        grep -oP '<a href="\K(ubuntu|rhel|sle|el)(?=/")' | \
        sort -u
}

# Get OS versions available for a specific ROCm version and OS type
get_os_versions_for_rocm_and_type() {
    local rocm_version="$1"
    local os_type="$2"
    local distro=$(distro_to_repo "$os_type")
    
    local versions=$(curl -s "$BASE_URL/$rocm_version/$os_type/" | \
        grep -oP '<a href="\K[^/"]+(?=/")' | \
        grep -v '^\.\.$' | \
        sort -V)
    
    # Convert Ubuntu codenames back to version numbers
    if [[ "$os_type" == "ubuntu" ]]; then
        while IFS= read -r ver; do
            os_version_from_repo "ubuntu" "$ver"
        done <<< "$versions"
    else
        echo "$versions"
    fi
}

# Check if a specific combination exists
check_combination_exists() {
    local rocm_version="$1"
    local os_type="$2"
    local os_version="$3"
    
    # Convert to repo format (handles Ubuntu codenames)
    local distro=$(distro_to_repo "$os_type")
    local repo_version=$(os_version_to_repo "$distro" "$os_version")
    
    # Try to fetch the directory, return 0 if HTTP 200, 1 otherwise
    local http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/$rocm_version/$os_type/$repo_version/")
    [[ "$http_code" == "200" ]]
}

# Mode 1: List ROCm versions for a given OS
list_rocm_for_os() {
    local repo_name=$(distro_to_repo "$DISTRO")
    
    # Normalize the OS version for display (convert codenames to version numbers)
    local display_version="$OS_VERSION"
    if [[ "$DISTRO" == "ubuntu" ]]; then
        display_version=$(ubuntu_codename_to_version "$OS_VERSION")
    fi
    
    echo -e "${BLUE}Checking ROCm versions available for $DISTRO $display_version...${NC}\n"
    
    local rocm_versions=$(get_rocm_versions)
    rocm_versions=$(filter_versions "$rocm_versions")
    
    local found_versions=()
    
    while IFS= read -r rocm_ver; do
        if check_combination_exists "$rocm_ver" "$repo_name" "$OS_VERSION"; then
            found_versions+=("$rocm_ver")
        fi
    done <<< "$rocm_versions"
    
    if [[ ${#found_versions[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No ROCm versions found for $DISTRO $display_version${NC}"
        return
    fi
    
    case "$OUTPUT_FORMAT" in
        matrix)
            echo -e "${GREEN}Supported ROCm versions:${NC}"
            printf '%s\n' "${found_versions[@]}" | column -c 80
            ;;
        list)
            printf '%s\n' "${found_versions[@]}"
            ;;
        yaml)
            echo "matrix:"
            for ver in "${found_versions[@]}"; do
                echo "  - os-distro: \"$DISTRO\""
                echo "    os-version: \"$display_version\""
                echo "    rocm-version: \"$ver\""
            done
            ;;
    esac
}

# Mode 2: List OS distributions for a given ROCm version
list_os_for_rocm() {
    local rocm_version="$1"
    
    echo -e "${BLUE}Checking OS distributions available for ROCm $rocm_version...${NC}\n"
    
    local os_types=$(get_os_types_for_rocm "$rocm_version")
    
    if [[ -z "$os_types" ]]; then
        echo -e "${YELLOW}No OS distributions found for ROCm $rocm_version${NC}"
        return
    fi
    
    declare -A os_map
    
    while IFS= read -r os_type; do
        local os_versions=$(get_os_versions_for_rocm_and_type "$rocm_version" "$os_type")
        local friendly_name=$(repo_to_distro "$os_type")
        
        while IFS= read -r os_ver; do
            if [[ -n "$os_ver" ]]; then
                os_map["$friendly_name:$os_ver"]=1
            fi
        done <<< "$os_versions"
    done <<< "$os_types"
    
    if [[ ${#os_map[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No OS versions found for ROCm $rocm_version${NC}"
        return
    fi
    
    case "$OUTPUT_FORMAT" in
        matrix)
            echo -e "${GREEN}Supported OS distributions:${NC}"
            for key in "${!os_map[@]}"; do
                IFS=':' read -r distro version <<< "$key"
                printf "  %-12s %s\n" "$distro" "$version"
            done | sort
            ;;
        list)
            for key in "${!os_map[@]}"; do
                echo "${key//:/ }"
            done | sort
            ;;
        yaml)
            echo "matrix:"
            for key in "${!os_map[@]}"; do
                IFS=':' read -r distro version <<< "$key"
                echo "  - os-distro: \"$distro\""
                echo "    os-version: \"$version\""
                echo "    rocm-version: \"$rocm_version\""
            done | sort
            ;;
    esac
}

# Generate containers.yml format
generate_containers_yaml() {
    if [[ "$OMIT_PATCH_VERSIONS" == "true" ]]; then
        echo -e "${BLUE}Generating containers.yml with ROCm >= ${ROCPROFILER_SYSTEMS_LOWER_BOUND} (omitting patch versions)...${NC}" >&2
    else
        echo -e "${BLUE}Generating containers.yml with ROCm >= ${ROCPROFILER_SYSTEMS_LOWER_BOUND} (including patch versions)...${NC}" >&2
    fi
    echo ""
    
    # Get all ROCm versions >= ROCPROFILER_SYSTEMS_LOWER_BOUND
    local rocm_versions=$(get_rocm_versions)
    rocm_versions=$(filter_versions "$rocm_versions")
    
    declare -A repo_map
    repo_map["ubuntu"]="ubuntu"
    repo_map["sle"]="opensuse"
    repo_map["rhel"]="rhel"
    
    declare -A os_combinations
    
    # Scan each ROCm version
    while IFS= read -r rocm_ver; do
        if [[ "$OMIT_PATCH_VERSIONS" == "true" ]]; then
            local version_parts=$(echo "$rocm_ver" | tr '.' ' ' | wc -w)
            if [[ $version_parts -gt 2 ]]; then
                continue  # Skip this version
            fi
        fi
        
        echo -e "${YELLOW}Scanning ROCm $rocm_ver...${NC}" >&2
        
        local os_types=$(get_os_types_for_rocm "$rocm_ver")
        
        while IFS= read -r os_type; do
            # Only process ubuntu, sle, rhel
            if [[ "$os_type" != "ubuntu" && "$os_type" != "sle" && "$os_type" != "rhel" ]]; then
                continue
            fi
            
            local os_versions=$(get_os_versions_for_rocm_and_type "$rocm_ver" "$os_type")
            local friendly_distro="${repo_map[$os_type]}"
            
            while IFS= read -r os_ver; do
                if [[ -n "$os_ver" ]]; then
                    local key="$friendly_distro:$os_ver"
                    if [[ -z "${os_combinations[$key]}" ]]; then
                        os_combinations[$key]="$rocm_ver"
                    else
                        os_combinations[$key]="${os_combinations[$key]} $rocm_ver"
                    fi
                fi
            done <<< "$os_versions"
        done <<< "$os_types"
    done <<< "$rocm_versions"
    
    # Output YAML header
    echo "# Supported OS + ROCm combinations for continuous integration"
    echo ""
    echo "matrix:"
    
    # Sort by distro and version, then output
    for key in $(printf '%s\n' "${!os_combinations[@]}" | sort); do
        IFS=':' read -r distro version <<< "$key"
        local rocm_list="${os_combinations[$key]}"
        
        echo "  # $distro $version"
        # First entry with rocm-version: "0.0"
        echo "  - os-distro: \"$distro\""
        echo "    os-version: \"$version\""
        echo "    rocm-version: \"0.0\""
        
        # Then entries for each supported ROCm version
        for rocm in $rocm_list; do
            echo "  - os-distro: \"$distro\""
            echo "    os-version: \"$version\""
            echo "    rocm-version: \"$rocm\""
        done
    done
}


main() {
    parse_args "$@"

    if [[ "$UPDATE_CONTAINERS" == "true" ]]; then
        # Mode: Generate containers.yml
        # Override LOWER_BOUND with ROCPROFILER_SYSTEMS_LOWER_BOUND
        LOWER_BOUND="$ROCPROFILER_SYSTEMS_LOWER_BOUND"
        generate_containers_yaml
    elif [[ -n "$DISTRO" && -n "$OS_VERSION" ]]; then
        # Mode 1: Query ROCm versions for specific OS
        list_rocm_for_os
    elif [[ -n "$ROCM_VERSION" ]]; then
        # Mode 2: Query OS distributions for specific ROCm version
        list_os_for_rocm "$ROCM_VERSION"
    else
        usage
    fi
}

main "$@"