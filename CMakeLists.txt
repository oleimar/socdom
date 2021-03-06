cmake_minimum_required(VERSION 3.10)

# The SocDomGen program runs evolutionary simulations
# Copyright (C) 2020  Olof Leimar
# See Readme.md for copyright notice

# set the project name
project(SocDomGen VERSION 1.0 LANGUAGES CXX C)

# set c++ standard
set(CMAKE_CXX_STANDARD 14)

# we need HighFive to handle hdf5 files
find_package(HighFive REQUIRED)

find_package(OpenMP)

# add the executable
add_executable(SocDomGen Evo.cpp EvoCode.cpp hdf5code.cpp Utils.cpp)

# link target
target_link_libraries(SocDomGen PUBLIC HighFive OpenMP::OpenMP_CXX)
