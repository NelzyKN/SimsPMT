#!/bin/bash
#
# ONE-TIME INSTALLER for Offline Setup
# Run this ONCE: bash install_offline_setup.sh
# Then use: setup_offline (from anywhere)

echo "════════════════════════════════════════════════════════════"
echo "     Installing Auger Offline Environment Setup"
echo "════════════════════════════════════════════════════════════"

# Create the setup script in home directory
cat > ~/.auger_offline_env << 'OFFLINE_ENV'
# Auger Offline Environment
# Auto-generated setup script - DO NOT EDIT MANUALLY

# Check if already loaded
if [ "$OFFLINE_SETUP_COMPLETE" = "1" ]; then
    echo "Offline environment already loaded."
    return 0
fi

echo "Loading Auger Offline environment..."

# Save current directory
OFFLINE_OLD_PWD=$(pwd)

# BASE PATHS
export APE_BASE=/home/kdnguyen/Downloads/ape-auger-v4r0p1-icrc2023-prod1
export AUGER_BASE=/afs/auger.mtu.edu/home/kdnguyen/auger/software/ApeInstall
export SYSTEM_APE=/afs/auger.mtu.edu/system/ubuntu-20.04/ape_trunks/Jun2022

# OFFLINE SETUP
eval $(${APE_BASE}/ape sh externals offline 2>/dev/null)
export AUGEROFFLINEROOT=${AUGER_BASE}/offline/4.0.1-icrc2023-prod1
export AUGEROFFLINECONFIG=${AUGEROFFLINEROOT}/share/auger-offline/config

if [ -f "${AUGEROFFLINEROOT}/bin/auger-offline-config" ]; then
    eval $(cd ${AUGEROFFLINEROOT}/bin/; ./auger-offline-config --env-sh 2>/dev/null)
fi

# LIBRARY PATHS
export LD_LIBRARY_PATH=${AUGEROFFLINEROOT}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/geant4/10.04.p01/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/clhep/2.4.0.4/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/xerces-c/3.2.3/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/boost/1_78_0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/hdf5/1.12.1/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/fftw/3.3.10/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/eigen/3.4.0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/External/aevread/aevread_v02r00p04/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/cdas/v6r4p0/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/fdeventlib/4.2.1/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${AUGER_BASE}/aerarootio/v00r16/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${SYSTEM_APE}/External/root/5.34.00-36f558c7d9/lib/root:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/afs/auger.mtu.edu/home/kdnguyen/auger/software/gsl-local/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/kdnguyen/miniconda3/lib:${LD_LIBRARY_PATH}

# COMPATIBILITY FIXES (silent)
G4LIB=${AUGER_BASE}/External/geant4/10.04.p01/lib
[ ! -f ${G4LIB}/libG4ptl.so.0 ] && ln -sf ${SYSTEM_APE}/External/geant4/10.07.p03/lib/libG4ptl.so.0 ${G4LIB}/libG4ptl.so.0 2>/dev/null
[ ! -f ${G4LIB}/libG4tasking.so ] && ln -sf ${SYSTEM_APE}/External/geant4/10.07.p03/lib/libG4tasking.so ${G4LIB}/libG4tasking.so 2>/dev/null

# PATH AND ENVIRONMENT
export PATH=${AUGEROFFLINEROOT}/bin:${PATH}
export GSL_DIR=/afs/auger.mtu.edu/home/kdnguyen/auger/software/gsl-local
export GSL_CONFIG=${GSL_DIR}/bin/gsl-config
export BOOST_ROOT=${AUGER_BASE}/External/boost/1_78_0
export G4INSTALL=${AUGER_BASE}/External/geant4/10.04.p01
export G4SYSTEM=Linux-g++
export CLHEP_BASE_DIR=${AUGER_BASE}/External/clhep/2.4.0.4

# UTILITIES
check_libs() {
    if [ -n "$1" ]; then
        ldd "$1" 2>/dev/null | grep -E "(not found|=>)"
    else
        echo "Usage: check_libs <program_or_library>"
    fi
}

rebuild_offline() {
    if [ -f "Makefile" ]; then
        echo "Rebuilding Offline program..."
        make clean && make
    else
        echo "Error: No Makefile in current directory"
        return 1
    fi
}

offline_help() {
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              Auger Offline Commands                       ║"
    echo "╠═══════════════════════════════════════════════════════════╣"
    echo "║  setup_offline    - Load Offline environment              ║"
    echo "║  check_libs       - Check library dependencies            ║"
    echo "║  rebuild_offline  - Rebuild program in current dir        ║"
    echo "║  offline_help     - Show this help message                ║"
    echo "╠═══════════════════════════════════════════════════════════╣"
    echo "║  AUGEROFFLINEROOT:                                        ║"
    echo "║    $AUGEROFFLINEROOT"
    echo "╚═══════════════════════════════════════════════════════════╝"
}

export -f check_libs rebuild_offline offline_help

# Restore directory
cd "$OFFLINE_OLD_PWD" 2>/dev/null
unset OFFLINE_OLD_PWD

# Success
echo "✅ Offline environment loaded successfully!"
echo "   Type 'offline_help' for available commands"

export OFFLINE_SETUP_COMPLETE=1
OFFLINE_ENV

# Add alias to .bashrc if not already there
if ! grep -q "alias setup_offline=" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Auger Offline Setup (added by installer)" >> ~/.bashrc
    echo "alias setup_offline='source ~/.auger_offline_env'" >> ~/.bashrc
    echo "✅ Added 'setup_offline' alias to ~/.bashrc"
else
    echo "ℹ️  Alias 'setup_offline' already exists in ~/.bashrc"
fi

# Also add to .bash_aliases if it exists
if [ -f ~/.bash_aliases ]; then
    if ! grep -q "alias setup_offline=" ~/.bash_aliases 2>/dev/null; then
        echo "alias setup_offline='source ~/.auger_offline_env'" >> ~/.bash_aliases
        echo "✅ Added 'setup_offline' alias to ~/.bash_aliases"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "                   Installation Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  To use the Offline environment, run:"
echo ""
echo "      setup_offline"
echo ""
echo "  This command is now available in all new terminals."
echo "  For the current terminal, run:"
echo ""
echo "      source ~/.bashrc"
echo "      setup_offline"
echo ""
echo "════════════════════════════════════════════════════════════"

# Offer to load it now
echo ""
read -p "Load Offline environment now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    source ~/.auger_offline_env
fi
