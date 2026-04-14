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

CMAKE_BIN="/opt/cmake-3.21.4/bin/cmake"

print_message() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (sudo)."
        exit 1
    fi
}

install_dependencies() {
    print_message "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y git cmake build-essential python3-dev wget python3-pybind11
    elif command -v pacman &> /dev/null; then
        pacman -Syu --noconfirm --needed git base-devel python python-pip pybind11 wget
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
        -DTFLITE_ENABLE_XNNPACK=OFF
    ${CMAKE_BIN} --build . -j $(nproc)
    if [ ! -f "${BUILD_DIR}/libtensorflow-lite.so" ]; then
        print_error "Build failed: libtensorflow-lite.so not found."
        exit 1
    fi
    print_message "Shared library built successfully."
}

install_library_and_headers() {
    print_message "Installing shared library to ${LIB_DEST} and all headers to ${INSTALL_PREFIX}/include..."

    # 1. Install the shared library
    cp "${BUILD_DIR}/libtensorflow-lite.so" "${LIB_DEST}/"
    chmod 755 "${LIB_DEST}/libtensorflow-lite.so"
    ldconfig   # update linker cache

    # 2. Remove any previous header installation to avoid conflicts
    rm -rf "${INSTALL_PREFIX}/include/tensorflow"

    # 3. Copy the entire tensorflow/ subdirectory from the source tree
    #    This includes all needed headers:
    #      - tensorflow/lite/          (core TFLite API)
    #      - tensorflow/compiler/      (required by interpreter.h)
    #      - tensorflow/core/          (internal dependencies)
    #      - tensorflow/tsl/           (support library)
    cp -r "${TFLITE_SRC_DIR}/tensorflow" "${INSTALL_PREFIX}/include/"

    # 4. Optional: also copy third-party headers if needed (e.g., flatbuffers, eigen)
    #    Usually they are not required for the public API, but if you get missing
    #    includes like "flatbuffers/flatbuffers.h", uncomment the following lines:
    # if [ -d "${TFLITE_SRC_DIR}/third_party" ]; then
    #     cp -r "${TFLITE_SRC_DIR}/third_party" "${INSTALL_PREFIX}/include/"
    # fi

    print_message "Installation complete."
    echo "Library: ${LIB_DEST}/libtensorflow-lite.so"
    echo "Headers: ${INSTALL_PREFIX}/include/tensorflow/"
}

cleanup() {
    print_message "Cleaning up temporary work directory: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
}

main() {
    check_root
    install_dependencies
    ensure_cmake
    check_cmake_version
    prepare_work_directory
    clone_tensorflow
    build_tflite_shared
    install_library_and_headers
    cleanup
    print_message "All done! You can now use the library in your C++ projects."
}

main "$@"
