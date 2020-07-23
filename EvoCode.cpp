#include "cpptoml.h"   // to read input parameters from TOML file
#include "EvoCode.hpp"
#include "hdf5code.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

#ifdef PARA_RUN
#include <omp.h>
#endif

// The SocDomGen program runs evolutionary simulations
// Copyright (C) 2020  Olof Leimar
// See Readme.md for copyright notice

//************************** Read and ReadArr ****************************

// convenience functions to read from TOML input file

// this template function can be used for any type of single value
template<typename T>
void Get(std::shared_ptr<cpptoml::table> infile,
         T& value, const std::string& name)
{
    auto val = infile->get_as<T>(name);
    if (val) {
        value = *val;
    } else {
        std::cerr << "Read failed for identifier " << name << "\n";
    }
}

// this template function can be used for a vector or array (but there is no
// checking how many elements are read)
template<typename It>
void GetArr(std::shared_ptr<cpptoml::table> infile,
            It beg, const std::string& name)
{
    using valtype = typename std::iterator_traits<It>::value_type;
    auto vp = infile->get_array_of<valtype>(name);
    if (vp) {
        std::copy(vp->begin(), vp->end(), beg);
    } else {
        std::cerr << "Read failed for identifier " << name << "\n";
    }
}


//************************** class EvoInpData ****************************

EvoInpData::EvoInpData(const char* filename) :
      OK(false)
{
    auto idat = cpptoml::parse_file(filename);
    Get(idat, max_num_thrds, "max_num_thrds");
    Get(idat, num_loci, "num_loci");
    Get(idat, nsp, "nsp");
    Get(idat, ngsp, "ngsp");
    Get(idat, gs, "gs");
    Get(idat, T, "T");
    Get(idat, tau, "tau");
    Get(idat, numby, "numby");
    Get(idat, numgen, "numgen");
    Get(idat, V0, "V0");
    Get(idat, V, "V");
    Get(idat, C, "C");
    Get(idat, a1, "a1");
    Get(idat, b1, "b1");
    Get(idat, b2, "b2");
    Get(idat, s1, "s1");
    Get(idat, s2, "s2");
    Get(idat, s3, "s3");
    Get(idat, sigma, "sigma");
    Get(idat, sigp, "sigp");
    Get(idat, mf1, "mf1");
    Get(idat, mf2, "mf2");
    Get(idat, pmax, "pmax");
    Get(idat, alphahat, "alphahat");
    Get(idat, betahat, "betahat");
    mut_rate.resize(num_loci);
    GetArr(idat, mut_rate.begin(), "mut_rate");
    SD.resize(num_loci);
    GetArr(idat, SD.begin(), "SD");
    max_val.resize(num_loci);
    GetArr(idat, max_val.begin(), "max_val");
    min_val.resize(num_loci);
    GetArr(idat, min_val.begin(), "min_val");
    rho.resize(num_loci);
    GetArr(idat, rho.begin(), "rho");
    Get(idat, learn_hist, "learn_hist");
    Get(idat, read_from_file, "read_from_file");
    cont_gen = false;
    if (read_from_file) {
        Get(idat, cont_gen, "cont_gen");
        Get(idat, h5InName, "h5InName");
    } else {
        all0.resize(num_loci);
        GetArr(idat, all0.begin(), "all0");
    }
    Get(idat, h5OutName, "h5OutName");
    if (learn_hist) {
        Get(idat, h5HistName, "h5HistName");
    }
    InpName = std::string(filename);
    OK = true;
}


//****************************** Class Evo *****************************

Evo::Evo(const EvoInpData& eid) :
    id{eid},
    num_loci{id.num_loci},
    nsp{id.nsp},
    ngsp{id.ngsp},
    gs{id.gs},
    ng{nsp*ngsp},
    Ns{ngsp*gs},
    N{ng*gs},
    T{id.T},
    tau{id.tau},
    numby{id.numby},
    numgen{id.numgen},
    V0{static_cast<flt>(id.V0)},
    V{static_cast<flt>(id.V)},
    C{static_cast<flt>(id.C)},
    a1{static_cast<flt>(id.a1)},
    b1{static_cast<flt>(id.b1)},
    b2{static_cast<flt>(id.b2)},
    s1{static_cast<flt>(id.s1)},
    s2{static_cast<flt>(id.s2)},
    s3{static_cast<flt>(id.s3)},
    sigma{static_cast<flt>(id.sigma)},
    sigp{static_cast<flt>(id.sigp)},
    mf1{static_cast<flt>(id.mf1)},
    mf2{static_cast<flt>(id.mf2)},
    pmax{static_cast<flt>(id.pmax)},
    alphahat{static_cast<flt>(id.alphahat)},
    betahat{static_cast<flt>(id.betahat)},
    learn_hist{id.learn_hist},
    num_thrds{1}
{
    // decide on number of threads for parallel processing
#ifdef PARA_RUN
    num_thrds = omp_get_max_threads();
    if (num_thrds > id.max_num_thrds) num_thrds = id.max_num_thrds;
    // check that there is at least one subpopulation per thread
    if (num_thrds > nsp) num_thrds = nsp;
    std::cout << "Number of threads: "
              << num_thrds << '\n';
#endif
    // generate one seed for each thread
    sds.resize(num_thrds);
    std::random_device rd;
    for (unsigned i = 0; i < num_thrds; ++i) {
        sds[i] = rd();
    }

    // Note concerning thread safety: in order to avoid possible problems with
    // multiple threads, the std::vector containers pop, next_pop, and stat are
    // allocated once and for all here, and thread-local data are then copied
    // into position in these (thus avoiding potentially unsafe push_back and
    // insert).

    // create N "placeholder individuals" in population
    gam_type gam(num_loci);
    ind_type indi(gs, gam);
    pop.resize(N, indi);
    next_pop.resize(N, indi);
    // learning history stats
    if (learn_hist) {
        stat.resize(nsp*ngsp*T);
    }

    // check if population data should be read from file
    if (id.read_from_file) {
        // Read_pop(id.InName);
        h5_read_pop(id.h5InName);
    } else {
        // construct all individuals as essentially the same
        gam_type gam(num_loci); // starting gamete
        for (unsigned l = 0; l < num_loci; ++l) {
            gam.gamdat[l] = static_cast<flt>(id.all0[l]);
        }
        unsigned j = 0;
        for (unsigned n = 0; n < nsp; ++n) { // subpopulations
            for (unsigned k = 0; k < ngsp; ++k) { // groups in subpopulation
                for (unsigned i = 0; i < gs; ++i) { // inds in group
                    ind_type ind(gs, gam);
                    ind.phenotype.spn = n; // set subpopulation number
                    ind.phenotype.gnum = n*ngsp + k; // set group number
                    ind.phenotype.Set_inum(i); // set individual number
                    pop[j++] = ind;
                }
            }
        }
    }
    // adjust number of bystanders, if needed
    if (numby > gs - 2) {
        if (gs - 2 < 0) {
            numby = 0;
        } else {
            numby = gs - 2;
        }
    }
}

void Evo::Run()
{
    Timer timer(std::cout);
    timer.Start();
    ProgressBar PrBar(std::cout, numgen);
#pragma omp parallel num_threads(num_thrds)
    {
#ifdef PARA_RUN
        int threadn = omp_get_thread_num();
#else
        int threadn = 0;
#endif
        // set up thread-local random number engine:
        rand_eng eng(sds[threadn]);
        rand_norm nr1(0, s1);
        rand_norm nr2(0, s2);
        rand_norm nr3(0, s3);
        rand_uni uni(0, 1);
        // set up thread-local mutation record, with thread-local engine and
        // parameters controlling mutation, segregation and recombination
        mut_rec_type mr(eng, num_loci);
        for (unsigned l = 0; l < num_loci; ++l) {
            mr.mut_rate[l] = static_cast<flt>(id.mut_rate[l]);
            mr.SD[l] = static_cast<flt>(id.SD[l]);
            mr.max_val[l] = static_cast<flt>(id.max_val[l]);
            mr.min_val[l] = static_cast<flt>(id.min_val[l]);
            mr.rho[l] = static_cast<flt>(id.rho[l]);
        }
        // determine which subpopulations this thread should handle
        unsigned num_per_thr = nsp/num_thrds;
        unsigned NP1 = threadn*num_per_thr;
        unsigned NP2 = NP1 + num_per_thr;
        if (threadn == num_thrds - 1) NP2 = nsp;
        // run through generations
        bool empi = false;
        for (unsigned gen = 0; gen < numgen; ++gen) {
            // get history only for final generation
            bool lhist = false;
            vs_type statl;
            if (gen == numgen - 1 && learn_hist) { // final generation
                statl.reserve((NP2 - NP1)*ngsp*T);
                lhist = true;
            }
            // set up (thread-local) vvind_type object
            vvind_type popl(NP2 - NP1);
            for (unsigned n = NP1; n < NP2; ++n) {
                vind_type& spl = popl[n - NP1];
                spl.reserve(Ns);
                // transfer individuals from global to thread-local
                for (unsigned i = 0; i < Ns; ++i) {
                    unsigned j = n*Ns + i;
                    spl.push_back(pop[j]);
                }
                if (gen > 0 || !id.cont_gen) {
                    // assign (random) quality values
                    unsigned i = 0;
                    for (unsigned k = 0; k < ngsp; ++k) {
                        // groups in subpopulation
                        for (unsigned j = 0; j < gs; ++j) { // inds in group
                            phen_type& ph = spl[i++].phenotype;
                            flt Z1 = nr1(eng);
                            flt Z2 = nr2(eng);
                            flt Z3 = nr3(eng);
                            ph.q = Z1;
                            ph.y = a1*Z1 + Z2;
                            ph.z = b1*Z1 + b2*Z2 + Z3;
                        }
                    }
                }
                // set up interaction groups, interact and get data
                for (unsigned k = 0; k < ngsp; ++k) {
                    vph_type phen(gs, phen_type(gs, gen_type(num_loci)));
                    for (unsigned j = 0; j < gs; ++j) {
                        unsigned i = k*gs + j;
                        phen_type& ph = phen[j];
                        ph = spl[i].phenotype;
                    }
                    unsigned gnum = n*ngsp + k;
                    if (gen == numgen - 1) {
                        // final generation, compute empirical score data
                        empi = true;
                    }
                    acg_type acg(gs, n, gnum, T, tau, V0, V, C, sigma, sigp,
                                 mf1, mf2, pmax, alphahat, betahat, phen,
                                 numby, lhist, empi);
                    acg.Interact(eng);
                    const vph_type& memb = acg.Get_memb();
                    for (unsigned j = 0; j < gs; ++j) {
                        unsigned i = k*gs + j;
                        spl[i].phenotype = memb[j];
                    }
                    if (gen == numgen - 1 && learn_hist) {
                        // final generation: append histories from the groups
                        // in this subpop
                        const vs_type& st = acg.Get_stat();
                        statl.insert(statl.end(), st.begin(), st.end());
                    }
                }
#pragma omp critical (set_next_pop)
                if (gen < numgen - 1) {
                    // if not final generation, get offspring from this
                    // subpopulation and put into next_pop
                    vind_type offs = SelectReproduce(spl, mr);
                    // copy individuals in offs to next_pop
                    for (unsigned i = 0; i < Ns; ++i) {
                        unsigned j = n*Ns + i;
                        next_pop[j] = offs[i];
                    }
                } else {
                    // final generation, copy individuals in spl to pop
                    for (unsigned i = 0; i < Ns; ++i) {
                        unsigned j = n*Ns + i;
                        pop[j] = spl[i];
                    }
                    if (learn_hist) {
                        // copy local history to global
                        for (unsigned k = 0; k < ngsp; ++k) {
                            for (unsigned l = 0; l < T; ++l) {
                                unsigned m1 = n*ngsp*T + k*T + l;
                                unsigned m2 = k*T + l;
                                stat[m1] = statl[m2];
                            }
                        }
                    }
                }
            }
#pragma omp barrier
            if (threadn == 0 && gen < numgen - 1) {
                // if not final generation, copy individuals from random
                // position in next_pop to pop, for start of next generation
                // (the code below uses randomly shuffled indices for this)
                ui_type indx(N, 0);
                for (unsigned j = 0; j < N; ++j) {
                    indx[j] = j;
                }
                // construct positions, 0 to N-1, of "random individuals"
                std::shuffle(indx.begin(), indx.end(), eng);
                // copy to pop
                for(unsigned j = 0; j < N; ++j) {
                    pop[j] = next_pop[indx[j]];
                    // set some numbering data for the individual
                    unsigned spn = j/Ns;
                    unsigned jj = j - spn*Ns;
                    unsigned gnum = jj/gs;
                    unsigned inum = jj - gnum*gs;
                    phen_type& ph = pop[j].phenotype;
                    ph.spn = spn;
                    ph.gnum = gnum;
                    ph.Set_inum(inum);
                }
                // all set to start next generation
                ++PrBar;
            }
        }
    }
    PrBar.Final();
    timer.Stop();
    timer.Display();
    h5_write_pop(id.h5OutName);
    if (learn_hist) {
        // Write_learn_hist();
        h5_write_hist(id.h5HistName);
    }
}

// return vector of Ns offspring from the subpopulation in sp, with individual
// payoff being proportional to the probability of delivering a gamete, and
// using mutation and recombination parameters from mr
Evo::vind_type Evo::SelectReproduce(const vind_type& sp, mut_rec_type& mr)
{
    vind_type offspr;
    offspr.reserve(Ns);
    unsigned np = sp.size(); // we ought to have np == Ns
    if (np > 0) {
        // get discrete distribution with parental payoffs as weights, taking
        // into account the costs of beta and u
        v_type wei(np);
        for (unsigned i = 0; i < np; ++i) {
            const phen_type& ph = sp[i].phenotype;
            wei[i] = ph.payoff;
            if (wei[i] < 0.0) {
                wei[i] = 0.0;
            }
        }
        rand_discr dscr(wei.begin(), wei.end());
        // get offspring
        for (unsigned j = 0; j < Ns; ++j) {
            // find "mother" for individual to be constructed
            unsigned imat = dscr(mr.eng);
            const ind_type& matind = sp[imat];
            // find "father" for individual to be constructed
            unsigned ipat = dscr(mr.eng);
            const ind_type& patind = sp[ipat];
            // append new individual to offspr
            offspr.emplace_back(gs, matind.GetGamete(mr),
                                patind.GetGamete(mr));
        }
    }
    return offspr;
}

void Evo::h5_read_pop(const std::string& infilename)
{
    // read data and put in pop
    h5R h5(infilename);
    std::vector<v_type> gams(N, v_type(num_loci));
    // read maternal gametes
    h5.read_flt_arr("MatGam", gams);
    for (unsigned i = 0; i < N; ++i) {
        gam_type& gam = pop[i].genotype.mat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = gams[i][l];
        }
    }
    // read paternal gametes
    h5.read_flt_arr("PatGam", gams);
    for (unsigned i = 0; i < N; ++i) {
        gam_type& gam = pop[i].genotype.pat_gam;
        for (unsigned l = 0; l < num_loci; ++l) {
            gam[l] = gams[i][l];
        }
    }
    v_type fval(N);
    // w0
    h5.read_flt("w0", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.w0 = fval[i];
    }
    // th0
    h5.read_flt("th0", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.th0 = fval[i];
    }
    // g0
    h5.read_flt("g0", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.g0 = fval[i];
    }
    // ga0
    h5.read_flt("ga0", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.ga0 = fval[i];
    }
    // alphw
    h5.read_flt("alphw", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.alphw = fval[i];
    }
    // alphth
    h5.read_flt("alphth", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.alphth = fval[i];
    }
    // beta
    h5.read_flt("beta", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.beta = fval[i];
    }
    // v
    h5.read_flt("v", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.v = fval[i];
    }
    // gf
    h5.read_flt("gf", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.gf = fval[i];
    }
    // q
    h5.read_flt("q", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.q = fval[i];
    }
    // y
    h5.read_flt("y", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.y = fval[i];
    }
    // z
    h5.read_flt("z", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.z = fval[i];
    }
    // payoff
    h5.read_flt("payoff", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.payoff = fval[i];
    }
    // EloR
    h5.read_flt("EloR", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.EloR = fval[i];
    }
    // std::vector to hold unsigned int member
    ui_type uival(N);
    // nInts
    h5.read_uint("nInts", uival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.nInts = uival[i];
    }
    // nAA
    h5.read_uint("nAA", uival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.nAA = uival[i];
    }
    // inum
    h5.read_uint("inum", uival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.inum = uival[i];
    }
    // gnum
    h5.read_uint("gnum", uival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.gnum = uival[i];
    }
    // spn
    h5.read_uint("spn", uival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.spn = uival[i];
    }
    // std::vector to hold int (actually bool) member
    std::vector<int> ival(N);
    // female
    h5.read_int("female", ival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.female = ival[i];
    }
    // alive
    h5.read_int("alive", ival);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.alive = ival[i];
    }
    // read learning parameters u and ga
    // u
    h5.read_flt("g", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.lp.g = fval[i];
    }
    // ga
    h5.read_flt("ga", fval);
    for (unsigned i = 0; i < N; ++i) {
        pop[i].phenotype.lp.ga = fval[i];
    }
    // read learning parameters w and th
    std::vector<v_type> pars(N, v_type(gs));
    // read w parameters
    h5.read_flt_arr("w", pars);
    for (unsigned i = 0; i < N; ++i) {
        v_type& par = pop[i].phenotype.lp.w;
        for (unsigned j = 0; j < gs; ++j) {
            par[j] = pars[i][j];
        }
    }
    // read th parameters
    h5.read_flt_arr("th", pars);
    for (unsigned i = 0; i < N; ++i) {
        v_type& par = pop[i].phenotype.lp.th;
        for (unsigned j = 0; j < gs; ++j) {
            par[j] = pars[i][j];
        }
    }
    // read lhat parameters
    h5.read_flt_arr("lhat", pars);
    for (unsigned i = 0; i < N; ++i) {
        v_type& par = pop[i].phenotype.lp.lhat;
        for (unsigned j = 0; j < gs; ++j) {
            par[j] = pars[i][j];
        }
    }
}

void Evo::h5_write_pop(const std::string& outfilename) const
{
    h5W h5(outfilename);
    std::vector<v_type> gams(N, v_type(num_loci));
    // write maternal gametes
    for (unsigned i = 0; i < N; ++i) {
        const gam_type& gam = pop[i].genotype.MatGam().Value();
        for (unsigned l = 0; l < num_loci; ++l) {
            gams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("MatGam", gams);
    // write paternal gametes
    for (unsigned i = 0; i < N; ++i) {
        const gam_type& gam = pop[i].genotype.PatGam().Value();
        for (unsigned l = 0; l < num_loci; ++l) {
            gams[i][l] = gam[l];
        }
    }
    h5.write_flt_arr("PatGam", gams);
    // write members of phenotypes
    // std::vector to hold flt member
    v_type fval(N);
    // w0
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.w0; });
    h5.write_flt("w0", fval);
    // th0
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.th0; });
    h5.write_flt("th0", fval);
    // g0
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.g0; });
    h5.write_flt("g0", fval);
    // ga0
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.ga0; });
    h5.write_flt("ga0", fval);
    // alphw
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.alphw; });
    h5.write_flt("alphw", fval);
    // alphth
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.alphth; });
    h5.write_flt("alphth", fval);
    // beta
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.beta; });
    h5.write_flt("beta", fval);
    // v
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.v; });
    h5.write_flt("v", fval);
    // gf
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.gf; });
    h5.write_flt("gf", fval);
    // q
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.q; });
    h5.write_flt("q", fval);
    // y
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.y; });
    h5.write_flt("y", fval);
    // z
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.z; });
    h5.write_flt("z", fval);
    // payoff
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.payoff; });
    h5.write_flt("payoff", fval);
    // EloR
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.EloR; });
    h5.write_flt("EloR", fval);
    // std::vector to hold unsigned int member
    ui_type uival(N);
    // nInts
    std::transform(pop.begin(), pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.nInts; });
    h5.write_uint("nInts", uival);
    // nAA
    std::transform(pop.begin(), pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.nAA; });
    h5.write_uint("nAA", uival);
    // inum
    std::transform(pop.begin(), pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.inum; });
    h5.write_uint("inum", uival);
    // gnum
    std::transform(pop.begin(), pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.gnum; });
    h5.write_uint("gnum", uival);
    // spn
    std::transform(pop.begin(), pop.end(), uival.begin(),
                   [](const ind_type& i) -> unsigned
                   { return i.phenotype.spn; });
    h5.write_uint("spn", uival);
    // std::vector to hold int (actually bool) member
    std::vector<int> ival(N);
    // female
    std::transform(pop.begin(), pop.end(), ival.begin(),
                   [](const ind_type& i) -> int
                   { return i.phenotype.female; });
    h5.write_int("female", ival);
    // alive
    std::transform(pop.begin(), pop.end(), ival.begin(),
                   [](const ind_type& i) -> int
                   { return i.phenotype.alive; });
    h5.write_int("alive", ival);
    // write learning parameters u and ga
    // g
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.lp.g; });
    h5.write_flt("g", fval);
    // ga
    std::transform(pop.begin(), pop.end(), fval.begin(),
                   [](const ind_type& i) -> flt
                   { return i.phenotype.lp.ga; });
    h5.write_flt("ga", fval);
    // write learning parameters w and th
    std::vector<v_type> pars(N, v_type(gs));
    // write w parameters
    for (unsigned i = 0; i < N; ++i) {
        const v_type& par = pop[i].phenotype.lp.w;
        for (unsigned j = 0; j < gs; ++j) {
            pars[i][j] = par[j];
        }
    }
    h5.write_flt_arr("w", pars);
    // write th parameters
    for (unsigned i = 0; i < N; ++i) {
        const v_type& par = pop[i].phenotype.lp.th;
        for (unsigned j = 0; j < gs; ++j) {
            pars[i][j] = par[j];
        }
    }
    h5.write_flt_arr("th", pars);
    // write lhat parameters
    for (unsigned i = 0; i < N; ++i) {
        const v_type& par = pop[i].phenotype.lp.lhat;
        for (unsigned j = 0; j < gs; ++j) {
            pars[i][j] = par[j];
        }
    }
    h5.write_flt_arr("lhat", pars);
}

void Evo::h5_write_hist(const std::string& histfilename) const
{
    h5W h5(histfilename);
    unsigned hlen = stat.size();
    // std::vector to hold unsigned int member
    ui_type uival(hlen);
    // spn
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.spn; });
    h5.write_uint("spn", uival);
    // gnum
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.gnum; });
    h5.write_uint("gnum", uival);
    // tstep
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.tstep; });
    h5.write_uint("tstep", uival);
    // i
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.i; });
    h5.write_uint("i", uival);
    // j
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.j; });
    h5.write_uint("j", uival);
    // ui
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.ui; });
    h5.write_uint("ui", uival);
    // uj
    std::transform(stat.begin(), stat.end(), uival.begin(),
                   [](const stat_type& st) -> unsigned
                   { return st.uj; });
    h5.write_uint("uj", uival);
    // std::vector to hold flt member
    v_type fval(hlen);
    // lij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.lij; });
    h5.write_flt("lij", fval);
    // lji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.lji; });
    h5.write_flt("lji", fval);
    // pij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.pij; });
    h5.write_flt("pij", fval);
    // pji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.pji; });
    h5.write_flt("pji", fval);
    // Rij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Rij; });
    h5.write_flt("Rij", fval);
    // Rji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Rji; });
    h5.write_flt("Rji", fval);
    // dij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.dij; });
    h5.write_flt("dij", fval);
    // dji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.dji; });
    h5.write_flt("dji", fval);
    // xi_ij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.xi_ij; });
    h5.write_flt("xi_ij", fval);
    // xi_ji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.xi_ji; });
    h5.write_flt("xi_ji", fval);
    // gi
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.gi; });
    h5.write_flt("gi", fval);
    // wii
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wii; });
    h5.write_flt("wii", fval);
    // wij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wij; });
    h5.write_flt("wij", fval);
    // gj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.gj; });
    h5.write_flt("gj", fval);
    // wjj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wjj; });
    h5.write_flt("wjj", fval);
    // wji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.wji; });
    h5.write_flt("wji", fval);
    // gai
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.gai; });
    h5.write_flt("gai", fval);
    // thii
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thii; });
    h5.write_flt("thii", fval);
    // thij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thij; });
    h5.write_flt("thij", fval);
    // gaj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.gaj; });
    h5.write_flt("gaj", fval);
    // thjj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thjj; });
    h5.write_flt("thjj", fval);
    // thji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.thji; });
    h5.write_flt("thji", fval);
    // lhatij
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.lhatij; });
    h5.write_flt("lhatij", fval);
    // lhatji
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.lhatji; });
    h5.write_flt("lhatji", fval);
    // EloRi
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.EloRi; });
    h5.write_flt("EloRi", fval);
    // EloRj
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.EloRj; });
    h5.write_flt("EloRj", fval);
}
