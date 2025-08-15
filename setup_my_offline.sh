#!/bin/bash
#
# Setup script for local Auger Offline installation
# Created for: kdnguyen's installation at /afs/auger.mtu.edu/home/kdnguyen/
# Offline version: 4.0.1-icrc2023-prod1
#
# Usage: source ~/setup_my_offline.sh
#        or: . ~/setup_my_offline.sh
#
# Note: This script must be sourced, not executed, to set environment variables

# Save current directory to return to it later
oldwd=$(pwd)

# ============================================================================
# BASE PATHS CONFIGURATION
# ============================================================================

# Your APE installer directory
export APE_BASE=/home/kdnguyen/Downloads/ape-auger-v4r0p1-icrc2023-prod1

# Your local installation base directory
export AUGER_BASE=/afs/auger.mtu.edu/home/kdnguyen/auger/software/ApeInstall

# System APE installation (for ROOT and other shared libraries)
export SYSTEM_APE=/afs/auger.mtu.edu/system/ubuntu-20.04/ape_trunks/Jun2022

# ============================================================================
# OFFLINE ENVIRONMENT SETUP
# ============================================================================

# Set up the environment for externals and offline using ape
echo "Setting up Offline environment..."
eval $(${APE_BASE}/ape sh externals offline 2>/dev/null)

# Set Offline-specific environment variables
export AUGEROFFLINEROOT=${AUGER_BASE}/offline/4.0.1-icrc2023-prod1
export AUGEROFFLINECONFIG=${AUGEROFFLINEROOT}/share/auger-offline/config

# Run the offline configuration script if it exists
if [ -f "${AUGEROFFLINEROOT}/bin/auger-offline-config" ]; then
    eval $(cd ${AUGEROFFLINEROOT}/bin/; ./auger-offline-config --env-sh 2>/dev/null)
else
    echo "Warning: auger-offline-config not found, setting basic paths..."
fi

# ============================================================================
# LIBRARY PATH CONFIGURATION
# ============================================================================

# Clear any duplicate paths and set up library paths in correct order

# 1. Your Offline libraries (highest priority)
export LD_LIBRARY_PATH=${AUGEROFFLINEROOT}/lib:${LD_LIBRARY_PATH}

# 2. YOUR local Geant4 and CLHEP (versions that Offline was built with)
export LD_LIBRARY_PATH=${AUGER_BASE}/External/geant4/10.04.p01/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/clhep/2.4.0.4/lib:${LD_LIBRARY_PATH}

# 3. Other external libraries from your installation
export LD_LIBRARY_PATH=${AUGER_BASE}/External/xerces-c/3.2.3/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/boost/1_78_0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/hdf5/1.12.1/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/fftw/3.3.10/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/eigen/3.4.0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/aevread/aevread_v02r00p04/lib:${LD_LIBRARY_PATH}

# 4. CDAS and other Auger-specific libraries
export LD_LIBRARY_PATH=${AUGER_BASE}/cdas/v6r4p0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/fdeventlib/4.2.1/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/aerarootio/v00r16/lib:${LD_LIBRARY_PATH}

# 5. ROOT libraries from system installation
export LD_LIBRARY_PATH=${SYSTEM_APE}/External/root/5.34.00-36f558c7d9/lib/root:${LD_LIBRARY_PATH}

# 6. Your local GSL installation
export LD_LIBRARY_PATH=/afs/auger.mtu.edu/home/kdnguyen/auger/software/gsl-local/lib:${LD_LIBRARY_PATH}

# 7. Conda/system libraries (lowest priority)
export LD_LIBRARY_PATH=/home/kdnguyen/miniconda3/lib:${LD_LIBRARY_PATH}

# ============================================================================
# COMPATIBILITY FIXES
# ============================================================================

# Create symbolic links for libraries that may be missing from older Geant4
# These are only created if they don't already exist

GEANT4_LIB_DIR=${AUGER_BASE}/External/geant4/10.04.p01/lib

# Check for libG4ptl.so.0 (added in later Geant4 versions)
if [ ! -f ${GEANT4_LIB_DIR}/libG4ptl.so.0 ] && [ -f ${SYSTEM_APE}/External/geant4/10.07.p03/lib/libG4ptl.so.0 ]; then
    echo "Note: Creating compatibility link for libG4ptl.so.0"
    ln -sf ${SYSTEM_APE}/External/geant4/10.07.p03/lib/libG4ptl.so.0 ${GEANT4_LIB_DIR}/libG4ptl.so.0 2>/dev/null
fi

# Check for libG4tasking.so (added in later Geant4 versions)
if [ ! -f ${GEANT4_LIB_DIR}/libG4tasking.so ] && [ -f ${SYSTEM_APE}/External/geant4/10.07.p03/lib/libG4tasking.so ]; then
    echo "Note: Creating compatibility link for libG4tasking.so"
    ln -sf ${SYSTEM_APE}/External/geant4/10.07.p03/lib/libG4tasking.so ${GEANT4_LIB_DIR}/libG4tasking.so 2>/dev/null
fi

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Add Offline binaries to PATH
export PATH=${AUGEROFFLINEROOT}/bin:${PATH}

# ============================================================================
# ADDITIONAL ENVIRONMENT VARIABLES
# ============================================================================

# Set up GSL environment
export GSL_DIR=/afs/auger.mtu.edu/home/kdnguyen/auger/software/gsl-local
export GSL_CONFIG=${GSL_DIR}/bin/gsl-config

# Set up Boost environment
export BOOST_ROOT=${AUGER_BASE}/External/boost/1_78_0

# Set up Geant4 environment
export G4INSTALL=${AUGER_BASE}/External/geant4/10.04.p01
export G4SYSTEM=Linux-g++

# Set up CLHEP environment
export CLHEP_BASE_DIR=${AUGER_BASE}/External/clhep/2.4.0.4

# ============================================================================
# VALIDATION AND STATUS
# ============================================================================

# Return to original directory
cd $oldwd

# Print status information
echo "=================================="
echo "Offline Environment Setup Complete"
echo "=================================="
echo "AUGEROFFLINEROOT: ${AUGEROFFLINEROOT}"
echo "Configuration:    ${AUGEROFFLINECONFIG}"
echo ""

# Check for key executables
if command -v EventFileReader &> /dev/null; then
    echo "âœ“ Offline executables found in PATH"
else
    echo "âš  Warning: Offline executables not found in PATH"
fi

# Check for auger-offline-config
if [ -f "${AUGEROFFLINEROOT}/bin/auger-offline-config" ]; then
    echo "âœ“ auger-offline-config available"
else
    echo "âš  Warning: auger-offline-config not found"
fi

echo ""
echo "Key libraries:"
echo "  Geant4: ${G4INSTALL}"
echo "  CLHEP:  ${CLHEP_BASE_DIR}"
echo "  Boost:  ${BOOST_ROOT}"
echo "  GSL:    ${GSL_DIR}"
echo ""
echo "To compile programs, use:"
echo "  make clean && make"
echo ""
echo "To run programs with XML config:"
echo "  ./userAugerOffline -b bootstrap.xml"
echo "=================================="

# Function to check library dependencies (optional utility)
check_libs() {
    if [ -n "$1" ]; then
        echo "Checking library dependencies for: $1"
        ldd $1 | grep -E "(not found|=>)" | head -20
    else
        echo "Usage: check_libs <executable_or_library>"
    fi
}

# Export the function for use in the shell
export -f check_libs
