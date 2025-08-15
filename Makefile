# Makefile for PMT Trace Module with ML Photon Trigger

# Source files - just add PhotonTriggerML.cc to the existing list
USER_SRCS := userAugerOffline.cc PMTTraceModule.cc PhotonTriggerML.cc

# XML configuration files
USER_XMLS := $(patsubst %.xml.in,%.xml,$(wildcard *.xml.in))

# Executable name
EXE := userAugerOffline

# Standard Auger Offline build rules
.PHONY: all depend clean run

ifdef AUGEROFFLINEROOT
  AUGEROFFLINECONFIG := $(AUGEROFFLINEROOT)/bin/auger-offline-config
else
  AUGEROFFLINECONFIG := auger-offline-config
  AUGEROFFLINEROOT := $(shell $(AUGEROFFLINECONFIG) --install)
endif

OBJS := $(USER_SRCS:.cc=.o)

CPPFLAGS    := $(shell $(AUGEROFFLINECONFIG) --cppflags)
CXXFLAGS    := $(shell $(AUGEROFFLINECONFIG) --cxxflags)
LDFLAGS_RAW := $(shell $(AUGEROFFLINECONFIG) --ldflags)
# Remove -lz from LDFLAGS
LDFLAGS     := $(filter-out -lz,$(LDFLAGS_RAW))
MAIN        := $(shell $(AUGEROFFLINECONFIG) --main)
CONFIGFILES := $(shell $(AUGEROFFLINECONFIG) --config)
XMLSCHEMALOCATION := $(shell $(AUGEROFFLINECONFIG) --schema-location)

all: $(EXE) $(USER_XMLS)

$(EXE): $(OBJS)
	@echo "Linking without -lz..."
	$(CXX) -o $@ $^ $(MAIN) $(CXXFLAGS) $(LDFLAGS)

%: %.in
	@echo -n "Generating $@ file... "
	@sed -e 's!@''CONFIGDIR@!$(CONFIGFILES)!g;s!@''SCHEMALOCATION@!$(XMLSCHEMALOCATION)!g;s!@''AUGEROFFLINEROOT@!$(AUGEROFFLINEROOT)!g' $< >$@
	@echo "done."

depend: Make-depend

Make-depend: $(USER_SRCS)
	$(CPP) $(CPPFLAGS) -MM $^ >$@

clean:
	- rm -f *.o $(EXE) $(USER_XMLS) Make-depend *.root

run: $(EXE) $(USER_XMLS)
	./$(EXE) -b bootstrap.xml

-include Make-depend
