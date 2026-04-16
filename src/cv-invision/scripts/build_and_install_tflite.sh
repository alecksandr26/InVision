TMP=`mktemp -t build_and_install_tflite.sh.XXXXXX`
trap "rm $TMP* 2>/dev/null" 0

#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TFLITE_VERSION="v2.18.0"
TFLITE_REPO_URL="https://github.com/tensorflow/tensorflow.git"
WORK_DIR="${HOME}/tensorflow_build_temp"
BUILD_DIR="${WORK_DIR}/build"
TFLITE_SRC_DIR="${WORK_DIR}/tensorflow"
INSTALL_PREFIX="/usr"
INCLUDE_DEST="${INSTALL_PREFIX}/include/tensorflow"
LIB_DEST="${INSTALL_PREFIX}/lib"
MANIFEST_DIR="/usr/share/tflite-custom"
MANIFEST_FILE="${MANIFEST_DIR}/manifest.txt"

CMAKE_BIN="/opt/cmake-3.21.4/bin/cmake"
KEEP_TEMP=false
UNINSTALL=false



print_message() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --keep-temp|-k) KEEP_TEMP=true; shift ;;
	    --uninstall)    UNINSTALL=true; shift ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (sudo)."
        exit 1
    fi
}


check_conflicts() {
    print_message "Checking for conflicting installations..."

    # Check system package
    if pacman -Q python-tensorflow &>/dev/null; then
        print_error "System package 'python-tensorflow' is installed. This may cause ABI conflicts."
        print_error "Please remove it with: sudo pacman -Rns python-tensorflow"
        exit 1
    fi

    # Check pip packages in the most common virtual environment locations
    # (This is heuristic; we can't know all venvs)
    if [ -f "env/bin/pip" ] && env/bin/pip list 2>/dev/null | grep -q "^tensorflow "; then
        print_warning "Virtual environment 'env' has 'tensorflow' pip package installed."
        print_warning "Consider uninstalling it: env/bin/pip uninstall tensorflow"
    fi

    if [ -f "env/bin/pip" ] && env/bin/pip list 2>/dev/null | grep -q "^tflite-runtime "; then
        print_warning "Virtual environment 'env' has 'tflite-runtime' pip package installed."
        print_warning "If you plan to use the C++ engine directly, this is fine. But avoid mixing Python TFLite."
    fi

    # Check NumPy version (optional, if you plan to install Python bindings later)
    if [ -f "env/bin/python" ]; then
        NUMPY_VER=$(env/bin/python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "none")
        if [[ "$NUMPY_VER" == 2.* ]]; then
            print_warning "NumPy $NUMPY_VER detected. tflite-runtime may not work with NumPy 2."
            print_warning "Downgrade with: env/bin/pip install 'numpy<2'"
        fi
    fi
}

record_installation() {
    mkdir -p "$(dirname "$MANIFEST_FILE")"
    > "$MANIFEST_FILE"   # truncate
    echo "# TFLite custom installation manifest" >> "$MANIFEST_FILE"
    echo "# Installed on $(date)" >> "$MANIFEST_FILE"
    echo "LIB:${LIB_DEST}/libtensorflow-lite.so" >> "$MANIFEST_FILE"

    # Record all dependency .so files we installed
    find "${BUILD_DIR}" -name "*.so" -type f ! -name "libtensorflow-lite.so" | while read -r so; do
        echo "DEPLIB:${LIB_DEST}/$(basename "$so")" >> "$MANIFEST_FILE"
    done

    # Record headers directory (recursive)
    echo "HEADERS:${INSTALL_PREFIX}/include/tensorflow" >> "$MANIFEST_FILE"

    print_message "Installation manifest saved to $MANIFEST_FILE"
}

uninstall() {
    if [ ! -f "$MANIFEST_FILE" ]; then
        print_error "Manifest file not found at $MANIFEST_FILE. Cannot uninstall."
        exit 1
    fi

    print_message "Uninstalling TFLite custom installation..."
    while IFS=: read -r type path; do
        if [[ "$type" == "LIB" && -f "$path" ]]; then
            rm -f "$path"
            print_message "Removed library: $path"
        elif [[ "$type" == "DEPLIB" && -f "$path" ]]; then
            rm -f "$path"
            print_message "Removed dependency library: $path"
        elif [[ "$type" == "HEADERS" && -d "$path" ]]; then
            rm -rf "$path"
            print_message "Removed headers: $path"
        fi
    done < "$MANIFEST_FILE"

    # Remove manifest directory if empty
    rm -f "$MANIFEST_FILE"
    rmdir "$(dirname "$MANIFEST_FILE")" 2>/dev/null || true

    # Reload linker cache
    ldconfig
    print_message "Uninstallation complete."
}

install_dependencies() {
    print_message "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y git cmake build-essential python3-dev wget python3-pybind11 flatbuffers-dev
    elif command -v pacman &> /dev/null; then
        pacman -Syu --noconfirm --needed git base-devel python python-pip pybind11 wget
        # flatbuffers - 24.3.x
        pacman -U https://archive.archlinux.org/packages/f/flatbuffers/flatbuffers-24.3.25-1-x86_64.pkg.tar.zst --noconfirm --needed
    else
        print_error "Unsupported distribution."
        exit 1
    fi
    print_message "Dependencies installed."
}



ensure_cmake() {
    if [ -f "${CMAKE_BIN}" ]; then
        print_message "CMake 3.21.4 already installed at ${CMAKE_BIN}"
        return 0
    fi

    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)
            CMAKE_ARCH="linux-x86_64"
            ;;
        aarch64)
            CMAKE_ARCH="linux-aarch64"
            ;;
        armv7l)
            CMAKE_ARCH="linux-armv7l"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH. Please install CMake manually."
            exit 1
            ;;
    esac

    print_message "Detected architecture: $ARCH. Downloading CMake for $CMAKE_ARCH..."

    local TARBALL="cmake-3.21.4-${CMAKE_ARCH}.tar.gz"
    local URL="https://github.com/Kitware/CMake/releases/download/v3.21.4/${TARBALL}"
    local DOWNLOAD_DIR="/tmp/cmake_install"
    
    mkdir -p "${DOWNLOAD_DIR}"
    cd "${DOWNLOAD_DIR}"
    
    print_message "Downloading ${URL} ..."
    wget --show-progress -q "${URL}" || {
        print_error "Failed to download CMake. Please check your internet connection."
        exit 1
    }
    
    print_message "Extracting ${TARBALL}..."
    tar -xzf "${TARBALL}" || {
        print_error "Failed to extract CMake tarball."
        exit 1
    }
    
    print_message "Moving to /opt/cmake-3.21.4..."
    mv "cmake-3.21.4-${CMAKE_ARCH}" /opt/cmake-3.21.4 || {
        print_error "Failed to move CMake to /opt. Do you have sudo permissions?"
        exit 1
    }
    
    cd /
    rm -rf "${DOWNLOAD_DIR}"
    
    print_message "CMake 3.21.4 installed successfully at ${CMAKE_BIN}"
}


check_cmake_version() {
    cmake_version=$(${CMAKE_BIN} --version | head -n1 | cut -d' ' -f3)
    required_version="3.16.0"
    if [[ "$(printf '%s\n' "$required_version" "$cmake_version" | sort -V | head -n1)" != "$required_version" ]]; then
        print_error "CMake version ${cmake_version} is less than required ${required_version}."
        exit 1
    fi
    print_message "CMake version ${cmake_version} meets the requirement."
}

prepare_work_directory() {
    print_message "Preparing work directory at ${WORK_DIR}..."
    if [ -d "${WORK_DIR}" ]; then
        print_warning "Directory ${WORK_DIR} already exists. Removing it."
        rm -rf "${WORK_DIR}"
    fi
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}"
}

clone_tensorflow() {
    print_message "Cloning TensorFlow repository (${TFLITE_VERSION})..."
    git clone --depth 1 --branch "${TFLITE_VERSION}" "${TFLITE_REPO_URL}" tensorflow
    if [ ! -d "${TFLITE_SRC_DIR}/tensorflow/lite" ]; then
        print_error "Cloning failed."
        exit 1
    fi
    print_message "Repository cloned successfully."
}

build_tflite_shared() {
    print_message "Building TensorFlow Lite as a shared library using ${CMAKE_BIN}..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    ${CMAKE_BIN} "${TFLITE_SRC_DIR}/tensorflow/lite" \
		 -DBUILD_SHARED_LIBS=ON \
		 -DTFLITE_ENABLE_XNNPACK=OFF \
		 -DTFLITE_STATIC_DEPS=ON \
		 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    ${CMAKE_BIN} --build . -j $(nproc)
    if [ ! -f "${BUILD_DIR}/libtensorflow-lite.so" ]; then
        print_error "Build failed: libtensorflow-lite.so not found."
        exit 1
    fi
    print_message "Shared library built successfully."
}

install_library_and_headers() {
    print_message "Installing shared library to ${LIB_DEST} and headers to ${INSTALL_PREFIX}/include..."

    # 1. Install the main TFLite shared library
    cp "${BUILD_DIR}/libtensorflow-lite.so" "${LIB_DEST}/"
    chmod 755 "${LIB_DEST}/libtensorflow-lite.so"

    # 2. Install dependency shared libraries
    print_message "Installing dependency shared libraries from build directory..."
    find "${BUILD_DIR}" -name "*.so" -type f | while read -r so; do
        if [[ "$so" == *"libtensorflow-lite.so"* ]]; then
            continue
        fi
        cp "$so" "${LIB_DEST}/"
        print_message "  Installed: $(basename "$so")"
    done

    # 3. Update linker cache
    ldconfig

    # 4. Install headers
    rm -rf "${INSTALL_PREFIX}/include/tensorflow"
    mkdir -p "${INSTALL_PREFIX}/include"
    cp -r "${TFLITE_SRC_DIR}/tensorflow" "${INSTALL_PREFIX}/include/"

    print_message "Installation complete."
}



cleanup() {
    if [ "$KEEP_TEMP" = false ]; then
        print_message "Cleaning up temporary work directory: ${WORK_DIR}"
        rm -rf "${WORK_DIR}"
    else
        print_message "Keeping temporary work directory as requested: ${WORK_DIR}"
    fi
}

main() {
    parse_args "$@"
    check_root


    if [ "$UNINSTALL" = true ]; then
        uninstall
        exit 0
    fi

    # Normal installation flow
    check_conflicts
    install_dependencies
    ensure_cmake
    check_cmake_version
    prepare_work_directory
    clone_tensorflow
    build_tflite_shared
    install_library_and_headers
    record_installation
    cleanup
    print_message "All done! You can now use the library in your C++ projects."
}

main "$@"
