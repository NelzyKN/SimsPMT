// userAugerOffline.cc
// Entry point for custom modules in Offline framework

#include <iostream>

// The MODULE_IS macro in PMTTraceModule.cc handles registration
// This function is called by Offline framework at startup
void AugerOfflineUser() 
{
    std::cout << "==================================" << std::endl;
    std::cout << "PMT Trace Extractor for CORSIKA Showers" << std::endl;
    std::cout << "Processing 0.1 EeV proton showers" << std::endl;
    std::cout << "==================================" << std::endl;
}
