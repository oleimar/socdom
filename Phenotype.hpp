#ifndef PHENOTYPE_HPP
#define PHENOTYPE_HPP

#include <string>
#include <array>
#include <ostream>
#include <istream>
#include <cmath>

// The SocDomGen program runs evolutionary simulations
// Copyright (C) 2020  Olof Leimar
// See Readme.md for copyright notice

//************************** struct LearnPars ******************************

// This struct stores an individual's current learning parameters, while
// learning from dominance interactions; the struct also stores the 'empirical
// logit scores' lhat

template<typename T>
struct LearnPars {
    using val_type = T;
    using par_type = std::vector<T>;
    using ui_type = std::vector<unsigned>;
    LearnPars(unsigned g_size,
              T w0 = 0, T th0 = 0) :
        w(g_size, w0),
        th(g_size, th0),
        lhat(g_size, 0),
        tlast(g_size, 0) {}
    T g;
    T ga;
    par_type w;
    par_type th;
    par_type lhat;
    ui_type tlast;
};


//************************* struct Phenotype ******************************

// Assumptions about GenType:
// types:
//   val_type
// member functions:
//   val_type Value()

// In addition to the genotypic trait values, consisting of w0, th0, g0, ga0,
// alphw, alphth, beta, v, gf, this class also stores the individual quality q,
// the 'perceived own fighting ability' y, the 'displayed fighting ability' z,
// the average per round payoff, the current 'Elo score', the current value of
// the group learning parameters (when saved this will be the value after the
// specified number of rounds of interaction during a generation), the number
// of rounds of interaction, individual number in group, the group number, the
// subpopulation number, the individual's sex, and whether it is alive.

template<typename GenType>
struct Phenotype {
// public:
    using flt = float;
    using v_type = std::vector<flt>;
    using lp_type = LearnPars<flt>;
    using gen_type = GenType;
    using val_type = typename gen_type::val_type;
    Phenotype(flt a_w0,
        flt a_th0,
        flt a_g0,
        flt a_ga0,
        flt a_alphw,
        flt a_alphth,
        flt a_beta,
        flt a_v,
        flt a_gf,
        flt a_q,
        flt a_y,
        flt a_z,
        flt a_payoff,
        unsigned g_size,
        unsigned a_inum,
        unsigned a_gnum,
        unsigned a_spn,
        bool a_female,
        bool a_alive) :
        w0{a_w0},
        th0{a_th0},
        g0{a_g0},
        ga0{a_ga0},
        alphw{a_alphw},
        alphth{a_alphth},
        beta{a_beta},
        v{a_v},
        gf{a_gf},
        q{a_q},
        y{a_y},
        z{a_z},
        payoff{a_payoff},
        lp(g_size, w0, th0),
        nInts{0},
        nAA{0},
        inum{a_inum},
        gnum{a_gnum},
        spn{a_spn},
        female{a_female},
        alive{a_alive} {}
    Phenotype(unsigned g_size = 0,
              const gen_type& gt = gen_type()) :
        lp(g_size) { Assign(gt); }
    void Assign(const gen_type& gt);
    void Set_inum(unsigned a_inum);
    // estimated value
    flt vhat(flt xi, unsigned j)
    {
        return gf*lp.w[inum] + (1 - gf)*lp.w[j] + lp.g*xi;
    }
    // logit of probability of choosing A (fight)
    flt l(flt xi, unsigned j)
    {
        return gf*lp.th[inum] + (1 - gf)*lp.th[j] + lp.ga*xi;
    }
    // probability of choosing A (fight)
    flt p(flt xi, unsigned j)
    {
        flt h = gf*lp.th[inum] + (1 - gf)*lp.th[j] + lp.ga*xi;
        return 1/(1 + std::exp(-h));
    }
    bool Female() const { return female; }
    // public data members
    flt w0;      // parameter for estimated reward at start of generation
    flt th0;     // parameter for preference at start of generation
    flt g0;      // parameter for estimated reward at start of generation
    flt ga0;     // parameter for preference at start of generation
    flt alphw;  // learning rate (salience)
    flt alphth; // learning rate (salience)
    flt beta;    // bystander learning rate factor
    flt v;       // perceived value of aggressive behaviour
    flt gf;      // generalisation factor
    flt q;       // individual quality (fighting ability)
    flt y;       // own perception of quality
    flt z;       // displayed quality
    flt payoff;  // the (fitness) payoff per round
    flt EloR;    // Elo rating
    lp_type lp;         // learning parameters
    unsigned nInts;     // number of interactions experienced
    unsigned nAA;       // number of AA interactions experienced
    unsigned inum;      // individual number
    unsigned gnum;      // group number
    unsigned spn;       // subpopulation number
    bool female;
    bool alive;
};

template<typename GenType>
void Phenotype<GenType>::Assign(const gen_type& gt)
{
    val_type val = gt.Value();
    // assume val is a vector with components corresponding to the traits w0,
    // th0, g0, ga0, alphw, alphth, beta, v, and gf
    w0 = val[0];
    th0 = val[1];
    g0 = val[2];
    ga0 = val[3];
    alphw = val[4];
    alphth = val[5];
    beta = val[6];
    v = val[7];
    gf = val[8];
    q = 0;
    y = 0;
    z = 0;
    // gf = 0;
    payoff = 0;
    EloR = 0;
    lp.g = g0;
    lp.ga = ga0;
    for (unsigned j = 0; j < lp.w.size(); ++j) {
        lp.w[j] = w0;
        lp.th[j] = th0;
        lp.lhat[j] = 0;
        lp.tlast[j] = 0;
    }
    nInts = 0;
    nAA = 0;
    inum = 0;
    gnum = 0;
    spn = 0;
    female = true;
    alive = true;
}

template<typename GenType>
void Phenotype<GenType>::Set_inum(unsigned a_inum)
{
    inum = a_inum;
}

#endif // PHENOTYPE_HPP
