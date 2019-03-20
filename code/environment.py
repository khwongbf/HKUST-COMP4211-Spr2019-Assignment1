# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:26:26 2019

Obtains the computing environment, except the modules.

@author: Wong Kwan Ho
"""

# Required imports for all tasks
from __future__ import print_function

# for presenting the computing environment
import platform
import sys
import subprocess

env_file = open("environment.txt", 'w+')

# Write the python version to the environment file
env_file.write("Python Version : " + sys.version + "\n")

# The platform information
env_file.write("Platform : " + platform.platform() + "\n")

# The architecture of the platform
env_file.write("Machine : " + str(platform.architecture()) + "\n")

# The processor
env_file.write("CPU : " + subprocess.getoutput(['wmic', 'cpu', 'get', 'name']) + "\n")

env_file.close()
