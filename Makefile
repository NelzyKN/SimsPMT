# Makefile for Auger Offline Modules with PhotonTriggerML
# Enhanced for ML-based photon trigger at T1 level

# ============================================================================
# Environment Variables (should be set by Auger Offline setup)
# ============================================================================
AUGEROFFLINE ?= $(shell echo $$AUGEROFFLINE)
AUGER_UTIL_DIR = $(AUGEROFFLINE)

# ============================================================================
# Compiler and Flags
# ============================================================================
CXX = g++
CXXFLAGS = -Wall -O2 -g -fPIC -std=c++11
LDFLAGS = -shared

# Add OpenMP support for parallel processing (optional)
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp

# ============================================================================
# Include Paths
# ============================================================================
INCLUDES = -I$(AUGEROFFLINE)/include \
           -I$(AUGEROFFLINE)/include/fwk \
           -I$(AUGEROFFLINE)/include/evt \
           -I$(AUGEROFFLINE)/include/sevt \
           -I$(AUGEROFFLINE)/include/sdet \
           -I$(AUGEROFFLINE)/include/det \
           -I$(AUGEROFFLINE)/include/utl \
           -I.

# ROOT configuration
ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)
ROOTGLIBS = $(shell root-config --glibs)

# Combine all includes
INCLUDES += $(ROOTCFLAGS)

# ============================================================================
# Library Paths and Libraries
# ============================================================================
LIBDIR = -L$(AUGEROFFLINE)/lib
LIBS = -lAugerFramework \
       -lAugerEvent \
       -lAugerSEvent \
       -lAugerDetector \
       -lAugerUtilities \
       -lAugerIO \
       $(ROOTLIBS) \
       -lMinuit \
       -lMathCore

# ============================================================================
# Source Files
# ============================================================================
# Original PMT Trace Module
PMT_TRACE_SOURCES = PMTTraceModule.cc
PMT_TRACE_HEADERS = PMTTraceModule.h
PMT_TRACE_OBJECTS = $(PMT_TRACE_SOURCES:.cc=.o)

# New ML Photon Trigger Module
PHOTON_ML_SOURCES = PhotonTriggerML.cc
PHOTON_ML_HEADERS = PhotonTriggerML.h
PHOTON_ML_OBJECTS = $(PHOTON_ML_SOURCES:.cc=.o)

# Combined sources
ALL_SOURCES = $(PMT_TRACE_SOURCES) $(PHOTON_ML_SOURCES)
ALL_HEADERS = $(PMT_TRACE_HEADERS) $(PHOTON_ML_HEADERS)
ALL_OBJECTS = $(PMT_TRACE_OBJECTS) $(PHOTON_ML_OBJECTS)

# ============================================================================
# Target Libraries
# ============================================================================
PMT_TRACE_LIB = libPMTTraceModule.so
PHOTON_ML_LIB = libPhotonTriggerML.so
ALL_LIBS = $(PMT_TRACE_LIB) $(PHOTON_ML_LIB)

# ============================================================================
# Build Rules
# ============================================================================
.PHONY: all clean install test debug help

# Default target - build all modules
all: $(ALL_LIBS)
	@echo "=== Build Complete ==="
	@echo "Built modules: $(ALL_LIBS)"

# Build PMTTraceModule library
$(PMT_TRACE_LIB): $(PMT_TRACE_OBJECTS)
	@echo "Building PMTTraceModule library..."
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)
	@echo "✓ PMTTraceModule built successfully"

# Build PhotonTriggerML library
$(PHOTON_ML_LIB): $(PHOTON_ML_OBJECTS)
	@echo "Building PhotonTriggerML library..."
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)
	@echo "✓ PhotonTriggerML built successfully"

# Compile PMTTraceModule
PMTTraceModule.o: PMTTraceModule.cc PMTTraceModule.h
	@echo "Compiling PMTTraceModule.cc..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile PhotonTriggerML
PhotonTriggerML.o: PhotonTriggerML.cc PhotonTriggerML.h
	@echo "Compiling PhotonTriggerML.cc..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Installation
# ============================================================================
INSTALL_DIR = $(AUGEROFFLINE)/userlib
INSTALL_INC_DIR = $(AUGEROFFLINE)/userinclude

install: $(ALL_LIBS)
	@echo "Installing modules to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)
	@mkdir -p $(INSTALL_INC_DIR)
	cp $(ALL_LIBS) $(INSTALL_DIR)/
	cp $(ALL_HEADERS) $(INSTALL_INC_DIR)/
	@echo "✓ Installation complete"
	@echo "Libraries installed to: $(INSTALL_DIR)"
	@echo "Headers installed to: $(INSTALL_INC_DIR)"

# ============================================================================
# Testing and Validation
# ============================================================================
# Test executable for standalone testing
TEST_EXEC = test_photon_trigger
TEST_SOURCES = test_photon_trigger.cc
TEST_OBJECTS = $(TEST_SOURCES:.cc=.o)

test: $(TEST_EXEC)
	@echo "Running tests..."
	./$(TEST_EXEC)

$(TEST_EXEC): $(TEST_OBJECTS) $(ALL_LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJECTS) -L. -lPhotonTriggerML -lPMTTraceModule $(LIBDIR) $(LIBS)

test_photon_trigger.o: test_photon_trigger.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Debug Build
# ============================================================================
debug: CXXFLAGS += -DDEBUG -g3 -O0
debug: clean all
	@echo "✓ Debug build complete"

# ============================================================================
# Performance Build (with optimization)
# ============================================================================
performance: CXXFLAGS += -O3 -march=native -mtune=native -DNDEBUG
performance: clean all
	@echo "✓ Performance build complete"

# ============================================================================
# Clean
# ============================================================================
clean:
	@echo "Cleaning build files..."
	rm -f $(ALL_OBJECTS) $(ALL_LIBS) $(TEST_OBJECTS) $(TEST_EXEC)
	rm -f *.o *.so *~ core *.root
	@echo "✓ Clean complete"

# ============================================================================
# Help
# ============================================================================
help:
	@echo "=========================================="
	@echo "Auger Offline Modules Makefile"
	@echo "=========================================="
	@echo "Targets:"
	@echo "  all         - Build all modules (default)"
	@echo "  install     - Install modules to AUGEROFFLINE"
	@echo "  test        - Build and run tests"
	@echo "  debug       - Build with debug symbols"
	@echo "  performance - Build with optimizations"
	@echo "  clean       - Remove build files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Modules:"
	@echo "  PMTTraceModule    - FADC trace extraction"
	@echo "  PhotonTriggerML   - ML-based photon trigger"
	@echo ""
	@echo "Environment:"
	@echo "  AUGEROFFLINE = $(AUGEROFFLINE)"
	@echo "==========================================

# ============================================================================
# Dependency Generation
# ============================================================================
depend: $(ALL_SOURCES)
	@echo "Generating dependencies..."
	$(CXX) $(INCLUDES) -MM $(ALL_SOURCES) > .depend

-include .depend

# ============================================================================
# Pattern Rules
# ============================================================================
%.o: %.cc %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.so: %.o
	$(CXX) $(LDFLAGS) -o $@ $< $(LIBDIR) $(LIBS)

# ============================================================================
# Validation of ML Model
# ============================================================================
validate-ml: $(PHOTON_ML_LIB)
	@echo "Validating ML photon trigger performance..."
	@echo "Running on test dataset..."
	AugerOffline -b bootstrap_test.xml
	@echo "Analyzing results..."
	root -l -q -b analyze_ml_performance.C
	@echo "✓ Validation complete - check photon_trigger_ml.root"

# ============================================================================
# FPGA Code Generation (Future)
# ============================================================================
fpga-gen: PhotonTriggerML.cc PhotonTriggerML.h
	@echo "Generating FPGA code (placeholder)..."
	@echo "Would convert 8-bit quantized NN to VHDL/Verilog"
	@echo "Target: Xilinx/Altera FPGA on AugerPrime boards"
	@echo "Not yet implemented"

# ============================================================================
# Documentation Generation
# ============================================================================
docs:
	@echo "Generating documentation..."
	doxygen Doxyfile
	@echo "✓ Documentation generated in doc/html/"
