#ifndef EVOCODE_HPP
#define EVOCODE_HPP

#ifdef _OPENMP
#define PARA_RUN
#endif

#include "Genotype.hpp"
#include "Phenotype.hpp"
#include "Individual.hpp"
#include "ACgroup.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

// The SocDomGen program runs evolutionary simulations
// Copyright (C) 2020  Olof Leimar
// See Readme.md for copyright notice

// An individual has 9 genetically determined traits, w0, th0, g0, ga0, alphw,
// alphth, beta, v, gf (see Phenotype.hpp), and there is one locus for each
// trait

//************************* Class EvoInpData ***************************

// This class is used to 'package' input data in a single place; the
// constructor extracts data from an input file

class EvoInpData {
public:
    using flt = double;         // the TOML class requires doubles
    using v_type = std::vector<flt>;
    std::size_t max_num_thrds;  // Max number of threads to use
    unsigned num_loci;          // Number of loci in individual's genotype
    unsigned nsp;               // Number of subpopulations
    unsigned ngsp;              // Number of groups per subpopulation
    unsigned gs;                // Group size
    unsigned T;                 // Number of rounds in group interaction
    unsigned tau;               // First round with chance of resource
    unsigned numgen;            // Number of generation to simulate
    unsigned numby;             // Number of bystanders to an interaction
    flt V0;                     // Baseline payoff per round
    flt V;                      // Resource payoff (in fitness terms)
    flt C;                      // Payoff cost parameter
    flt a1;                     // Parameter for 'own perceived quality'
    flt b1;                     // Parameter for 'displayed quality'
    flt b2;                     // Parameter for 'displayed quality'
    flt s1;                     // Within-group SD of quality distribution
    flt s2;                     // SD of increment for 'own perceived quality'
    flt s3;                     // SD of increment for 'displayed quality'
    flt sigma;                  // Parameter for 'relative quality' obs
    flt sigp;                   // Parameter for 'penalty of AA interaction'
    flt mf1;                    // Memory factor
    flt mf2;                    // Memory factor
    flt pmax;                   // Maximum value for probability to use A
    flt alphahat;               // parameter for empirical updates
    flt betahat;                // parameter for empirical bystander updates
    v_type mut_rate;            // Probability of mutation at each locus
    v_type SD;                  // SD of mutational increments at each locus
    v_type max_val;             // Maximal allelic value at each locus
    v_type min_val;             // Minimal allelic value at each locus
    v_type rho;                 // Recombination rates
    v_type all0;                // Starting allelic values (if not from file)
    bool learn_hist;            // Whether to compute and save learning history
    bool read_from_file;        // Whether to read population from file
    bool cont_gen;              // Whether to continue in the first generation
    std::string h5InName;       // File name for input of population
    std::string h5OutName;      // File name for output of population
    std::string h5HistName;     // File name for output of learning history

    std::string InpName;  // Name of input data file
    bool OK;              // Whether input data has been successfully read

    EvoInpData(const char* filename);
};


//***************************** Class Evo ******************************

class Evo {
public:
    // types needed to define individual
    using mut_rec_type = MutRec<MutIncrNorm<>>;
    using gam_type = Gamete<mut_rec_type>;
    using gen_type = Diplotype<gam_type>;
    using phen_type = Phenotype<gen_type>;
    using ind_type = Individual<gen_type, phen_type>;
    using stat_type = LearnStat<phen_type>;
    // use std::vector containers for (sub)populations
    using vind_type = std::vector<ind_type>;
    using vvind_type = std::vector<vind_type>;
    using vph_type = std::vector<phen_type>;
    using acg_type = ActCritGroup<phen_type>;
    using flt = float;
    using v_type = std::vector<flt>;
    using vs_type = std::vector<stat_type>;
    using ui_type = std::vector<unsigned>;
    using rand_eng = std::mt19937;
    // using rand_int = std::uniform_int_distribution<int>;
    using rand_uni = std::uniform_real_distribution<flt>;
    using rand_norm = std::normal_distribution<flt>;
    using rand_discr = std::discrete_distribution<int>;
    Evo(const EvoInpData& eid);
    void Run();
    void h5_read_pop(const std::string& infilename);
    void h5_write_pop(const std::string& outfilename) const;
    void h5_write_hist(const std::string& histfilename) const;
private:
    vind_type SelectReproduce(const vind_type& sp, mut_rec_type& mr);

    EvoInpData id;
    unsigned num_loci;
    unsigned nsp;
    unsigned ngsp;
    unsigned gs;
    unsigned ng;
    unsigned Ns;
    unsigned N;
    unsigned T;
    unsigned tau;
    unsigned numby;
    unsigned numgen;
    flt V0;
    flt V;
    flt C;
    flt a1;
    flt b1;
    flt b2;
    flt s1;
    flt s2;
    flt s3;
    flt sigma;
    flt sigp;
    flt mf1;
    flt mf2;
    flt pmax;
    flt alphahat;
    flt betahat;
    bool learn_hist;
    std::size_t num_thrds;
    ui_type sds;
    vind_type pop;
    vind_type next_pop;
    vs_type stat;
};

#endif // EVOCODE_HPP
