#!/usr/bin/env python3

"""
Balanced interleaving of proton and photon CORSIKA files
Ensures equal representation for ML training
"""

import os
import glob
import random
import sys

def find_corsika_files(base_path, pattern="DAT*"):
    """Find all CORSIKA DAT files in a directory"""
    files = []
    search_pattern = os.path.join(base_path, pattern)
    
    for f in glob.glob(search_pattern):
        # Exclude files with extensions (like DAT*.gz)
        if '.' not in os.path.basename(f).split('DAT')[1]:
            files.append(f)
    
    return files

def balanced_interleave(proton_files, photon_files, ratio=1.0):
    """
    Interleave proton and photon files with specified ratio
    ratio = 1.0 means equal numbers
    ratio = 2.0 means 2 protons per photon
    ratio = 0.5 means 2 photons per proton
    """
    
    # Shuffle both lists independently
    random.shuffle(proton_files)
    random.shuffle(photon_files)
    
    mixed_files = []
    
    # Determine how to interleave based on ratio
    if ratio == 1.0:
        # Equal interleaving
        min_len = min(len(proton_files), len(photon_files))
        for i in range(min_len):
            mixed_files.append(proton_files[i])
            mixed_files.append(photon_files[i])
        
        # Add remaining files
        if len(proton_files) > min_len:
            mixed_files.extend(proton_files[min_len:])
        if len(photon_files) > min_len:
            mixed_files.extend(photon_files[min_len:])
    
    else:
        # Custom ratio interleaving
        proton_idx = 0
        photon_idx = 0
        
        while proton_idx < len(proton_files) or photon_idx < len(photon_files):
            # Add protons based on ratio
            protons_to_add = int(ratio) if ratio >= 1 else 1
            for _ in range(protons_to_add):
                if proton_idx < len(proton_files):
                    mixed_files.append(proton_files[proton_idx])
                    proton_idx += 1
            
            # Add photons based on ratio
            photons_to_add = 1 if ratio >= 1 else int(1/ratio)
            for _ in range(photons_to_add):
                if photon_idx < len(photon_files):
                    mixed_files.append(photon_files[photon_idx])
                    photon_idx += 1
    
    return mixed_files

def main():
    # Paths to data
    proton_path = "/afs/auger.mtu.edu/common/data/pauger_SIM_SHOWERS/protons/corsika_77420_Auger_lib/SibyllStar/18.0_18.5/proton/"
    photon_path = "/afs/auger.mtu.edu/common/data/pauger_SIM_SHOWERS/photons/corsika_77420_NapoliPraha/EPOS_LHC/photon/18.0_18.5/run01/"
    
    # Find files
    print("Searching for proton files...")
    proton_files = find_corsika_files(proton_path)
    print(f"Found {len(proton_files)} proton files")
    
    print("Searching for photon files...")
    photon_files = find_corsika_files(photon_path)
    print(f"Found {len(photon_files)} photon files")
    
    # Balance the data - you can adjust the ratio
    # Since photons are rarer in reality, you might want to oversample them
    # ratio = 1.0 means equal representation (good for initial training)
    ratio = 1.0
    
    print(f"\nInterleaving with ratio {ratio} (protons:photons)")
    mixed_files = balanced_interleave(proton_files, photon_files, ratio)
    
    # Optional: Apply final shuffle for extra randomization
    # random.shuffle(mixed_files)
    
    # Write output
    output_file = "balanced_corsika_files.txt"
    with open(output_file, 'w') as f:
        for filepath in mixed_files:
            f.write(filepath + '\n')
    
    print(f"\nWrote {len(mixed_files)} files to {output_file}")
    
    # Statistics
    proton_count = sum(1 for f in mixed_files if 'proton' in f)
    photon_count = sum(1 for f in mixed_files if 'photon' in f)
    print(f"Final mix: {proton_count} protons, {photon_count} photons")
    print(f"Actual ratio: {proton_count/max(photon_count,1):.2f}:1")
    
    # Show sample of output
    print("\nFirst 10 files in mixed list:")
    for f in mixed_files[:10]:
        particle_type = "photon" if "photon" in f else "proton"
        print(f"  [{particle_type}] {os.path.basename(f)}")

if __name__ == "__main__":
    main()
