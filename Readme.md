
# socdom: C++ code for evolutionary simulation of dominance hierarchy formation


## Overview

This repository contains C++ code and example data.
The executable program `SocDomGen`, built from this code, will run evolutionary simulations of a population of several groups of individuals that, in each generation, have social dominance interactions.
The program was used to produce the evolutionary simulation results for the paper "The evolution of social dominance through reinforcement learning" by Olof Leimar.


## System requirements

This program has been compiled and run on a Linux server with Ubuntu 18.04 LTS.
The C++ compiler was g++ version 7.5.0, provided by Ubuntu, with compiler flags for c++14, and `cmake` (<https://cmake.org/>) was used to build the program.
It can be run multithreaded using OpenMP, which speeds up execution times.
Most likely the instructions below will work for many Linux distributions.
For single-threaded use, the program has also been compiled and run on macOS, using the Apple supplied Clang version of g++, but multithreaded use on macOS might be unreliable.

The program reads input parameters from TOML files (<https://github.com/toml-lang/toml>), using the open source `cpptoml.h` header file (<https://github.com/skystrife/cpptoml>), which is included in this repository.

The program stores evolving populations in HDF5 files (<https://www.hdfgroup.org/>), which is an open source binary file format.
The program uses the open source HighFive library (<https://github.com/BlueBrain/HighFive>) to read and write to such files.
These pieces of software need to be installed in order for `cmake` to successfully build the program.


## Installation guide

Install the repository from Github to a local computer.
There is a single directory `socdom` for source code and executable, a subdirectory `Data` where input data and data files containing simulated populations are kept, and a subdirectory `build` used by `cmake` for files generated during building, including the executable `SocDomGen`.


## Building the program

The CMake build system is used.
If it does not exist, create a build subdirectory in the project folder (`mkdir build`) and make it the current directory (`cd build`).
If desired, for a build from scratch, delete any previous content (`rm -rf *`).
Run CMake from the build directory. For a release build:
```
cmake -D CMAKE_BUILD_TYPE=Release ../
```
and for a debug build replace Release with Debug.
If this succeeds, i.e. if the `CMakeLists.txt` file in the project folder is processed without problems, build the program:
```
make
```
This should produce an executable in the `build` directory.


## Running

Make the Data directory current.
Assuming that the executable is called `SocDomGen` and with an input file called `Run11b.toml`, run the program as
```
../build/SocDomGen Run11b.toml
```
Alternatively, using an R script file `Run11b_run.R`, run the script as
```
Rscript Run11b_run.R
```
where `Rscript` is the app for running R scripts.
You need to have `R` installed for this to work.


## Description of the evolutionary simulations

There is an input file, for instance `Run11b.toml`, for each case, which typically simulates 5,000 generations, inputting the populations from, e.g., the HDF5 file `Run11b.h5` and outputting to the same file.
Without an existing `Run11b.h5` data file, the program can start by constructing individuals with genotypes fro the allelic values given by `all0`in the input file.
To make this happen, use `read_from_file = false` in the input file.
There is an R script, e.g. `Run11b_run.R`, which repeats such runs a number of times, typically 25, and for each run computes statistics on the evolving leaning traits and adds a row to a TSV data file, e.g. `Run11b_data.tsv`.

Each case is first run for many generations with higher mutation rates and then for many generation rates with lower mutation rates. When a seeming evolutionary equilibrium has been reached, 100 runs are kept in the summary data file, like `Run11b_data.tsv`. These then represent evolution over `100*5,000 = 500,000` generations. The mean of all these means, for each learning trait, is then used for simulations of dominance hierarchy formation for groups of individuals that are adapted to a particular situation.

### Basic cases

In order to give some focus and a starting point, there is a sequence of three cases: `Run11b`, `Run21b`, and `Run31b`, where `b` is for bystander. These correspond to group sizes `gs` of 4, 8, and 16 individuals over time steps `T` of 400, 800, and 1600 per generation (there is one dominance interaction per time step), with evolution of all learning parameters, including the generalisation parameter `f` and the rate `beta` of bystander learning, and with high memory factors (`mf1 = mf2 = 0.999`). Other parameters are at their 'standard' values: `V0 = 0.5, V = 0.25, C =  0.2, a1 = 0.707, b1 = 0.707, b2 = 0.0, s1 = 0.5, s2 = 0.01, s3 = 0.01, sigma = 0.5, mf1 = 0.999, mf2 = 0.999, pmax = 0.99`.

Concerning the observations of fighting ability, in the initial phase of each dominance interaction, these parameters (`a1 = 0.707, b1 = 0.707, b2 = 0.0, s1 = 0.5, s2 = 0.01, s3 = 0.01, sigma = 0.5`) imply that around 50% of the variation in the observations is due to variation in `qi - qj`. The parameter `pmax = 0.99` sets the exploration, and means that the probability of choosing A is never higher than 0.99, and never lower than 1 - 0.99 = 0.01.

These can be regarded as cases of individuals with high social competence, adapted to different group sizes.

### Brief description of other cases

Here are a number of cases that can be run in the same way as the basic cases:

* `Run11bg`, `Run21bg`, `Run31bg`. Same as `Run11b`, `Run21b`, `Run31b` except that the generalisation parameter is kept fixed at `f = 0.5`.

* `Run11nb`, `Run21nb`, `Run31nb`. Same as `Run11b`, `Run21b`, `Run31b` except that the bystander learning rate is kept fixed at `beta = 0.0`, so there are no bystander effects.

* `Run11nbg`, `Run21nbg`, `Run31nbg`. Same as `Run11b`, `Run21b`, `Run31b` except that the generalisation parameter is kept fixed at `f = 0.5` and the bystander learning rate is kept fixed at `beta = 0.0`. These can be regarded as cases of individuals with lower social competence, adapted to different group sizes.

* `Run11nbgf`, `Run21nbgf`, `Run31nbgf`. Same as `Run11b`, `Run21b`, `Run31b` except that the generalisation parameter is kept fixed at `f = 1.0`, so there is full generalisation (all opponents are treated the same) and the bystander learning rate is kept fixed at `beta = 0.0`. These can be regarded as cases of individuals without social competence, but still adapted to different group sizes.

* `Run11bo`, `Run21bo`, `Run31bo`. Same as `Run11b`, `Run21b`, `Run31b` except that `a1 = 0.25, b1 = 0.25`. This means that for these cases only around 11% of the variation in the observations is due to variation in `qi - qj`.


## License

The `SocDomGen` program runs evolutionary simulations of social hierarchy formation.

Copyright (C) 2020  Olof Leimar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

