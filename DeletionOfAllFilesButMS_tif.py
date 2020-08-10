"""
Created: 2020/14/03
Author: Jari
"""

#####
# Script for deletion of unwanted files in a given folder and subfolders
#
#
# WARNING: This Script deletes all files in a given folder and subfolders which do not end with a specific filename.
# In this case all files which do not end with 'MS.tif' will be deleted.
#
#####

import os

directory = r"F:\Geoinformatik\Planetscope Hunsrück\Planet\Planet_Daten"   #path to folder where files should be deleted
print("Starting deletion")
for root, dirs, files in os.walk(os.path.abspath(directory)):
    for file in files:
        if (os.path.join(root, file).endswith('MS.tif')):
            print("file kept: " + file)
        else:
            #print(os.path.join(root, file))
            os.remove(os.path.join(root, file))
            print("deleted: " + file)
        #print(os.path.join(root, file))
print("Deletion completed")