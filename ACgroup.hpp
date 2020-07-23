#ifndef ACGROUP_HPP
#define ACGROUP_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// The SocDomGen program runs evolutionary simulations
// Copyright (C) 2020  Olof Leimar
// See Readme.md for copyright notice

//************************* struct LearnStat ******************************

// This struct stores data on a dominance interaction between two group members
// (i and j); a sequence of structs can be saved as a record of the interaction
// history in the group

template<typename PhenType>
struct LearnStat {
    using phen_type = PhenType;
    using flt = typename phen_type::flt;
    unsigned spn;       // subpopulation number
    unsigned gnum;      // group number
    unsigned tstep;     // time step of interaction in group
    unsigned i;         // inum for individual
    unsigned j;         // inum for individual
    unsigned ui;        // action (1 is A)
    unsigned uj;        // action
    flt lij;            // logit of pij
    flt lji;            // logit of pji
    flt pij;            // prob for i to use action A
    flt pji;            // prob for j to use action A
    flt Rij;            // perceived reward by i
    flt Rji;            // perceived reward by j
    flt dij;            // TD error for i
    flt dji;            // TD error for j
    flt xi_ij;          // observation by i
    flt xi_ji;          // observation by j
    flt gi;             // slope parameter for i
    flt wii;            // intercept parameter for i
    flt wij;            // intercept parameter for i
    flt gj;             // slope parameter for j
    flt wjj;            // intercept parameter for j
    flt wji;            // intercept parameter for j
    flt gai;            // slope parameter for i
    flt thii;           // intercept parameter for i
    flt thij;           // intercept parameter for i
    flt gaj;            // slope parameter for j
    flt thjj;           // intercept parameter for j
    flt thji;           // intercept parameter for j
    flt lhatij;         // empirical logit score for i
    flt lhatji;         // empirical logit score for j
    flt EloRi;          // Elo score for i
    flt EloRj;          // Elo score for j
};


//************************ class ActCritGroup *****************************

// This class sets up and simulates the actor-critic learning method for a
// social dominance game.

// The class deals with the interactions in one group, over the time steps
// during one generation.

template<typename PhenType>
class ActCritGroup {
public:
    using phen_type = PhenType;
    using lp_type = typename phen_type::lp_type;
    using vph_type = std::vector<phen_type>;
    using stat_type = LearnStat<PhenType>;
    using vs_type = std::vector<stat_type>;
    using flt = typename phen_type::flt;
    using v_type = std::vector<flt>;
    using rand_eng = std::mt19937;
    using rand_uni = std::uniform_real_distribution<flt>;
    using rand_int = std::uniform_int_distribution<unsigned>;
    using rand_norm = std::normal_distribution<flt>;
    ActCritGroup(unsigned a_gs,
                 unsigned a_spn,
                 unsigned a_gnum,
                 unsigned a_T,
                 unsigned a_tau,
                 flt a_V0,
                 flt a_V,
                 flt a_C,
                 flt a_sigma,
                 flt a_sigp,
                 flt a_mf1,
                 flt a_mf2,
                 flt a_pmax,
                 flt a_alphahat,
                 flt a_betahat,
                 const vph_type& a_memb,
                 unsigned a_numby,
                 bool a_lhist = false,
                 bool a_empi = false);
    const vph_type& Get_memb() const { return memb; }
    const vs_type& Get_stat() const { return stat; }
    void Interact(rand_eng& eng);

private:
    void Forget(unsigned i, unsigned j, unsigned tstep);
    // value of perceived penalty in AA interaction between i and j
    flt ce(flt qi, flt qj, flt eq) { return C*std::exp(-qi + qj + eq); }
    flt Clamp(flt p)
        {return (p > pmax) ? pmax : ((p < 1 - pmax) ? 1 - pmax : p); }
    flt R(unsigned i, unsigned j, unsigned ui, unsigned uj, flt eq);
    void Update_payoffs(unsigned i, unsigned j,
                        unsigned ui, unsigned uj, flt val);
    void Update_w(unsigned i, unsigned j, flt xi_ij, flt deltaij);
    void Update_theta(unsigned i, unsigned j,
                      flt xi_ij, flt derij, flt deltaij);
    void Add_lstat(unsigned tstep,
                   unsigned i, unsigned j,
                   unsigned ui, unsigned uj,
                   flt xi_ij, flt xi_ji,
                   flt lij, flt lji,
                   flt pij, flt pji,
                   flt Rij, flt Rji,
                   flt deltaij, flt deltaji);
    unsigned gs;     // group size
    unsigned spn;    // subpopulation number
    unsigned gnum;   // group number
    unsigned T;      // total number of rounds for group
    unsigned tau;    // first round with chance for resource
    flt V0;          // payoff parameter
    flt V;           // payoff parameter
    flt C;           // payoff parameter
    flt sigma;       // SD error parameter for 'relative quality' observation
    flt sigp;        // SD error parameter for 'penalty of AA interaction'
    flt mf1;         // memory factor
    flt mf2;         // memory factor
    flt pmax;        // maximum value for probability to use A
    flt alphahat;    // parameter for empirical updates
    flt betahat;     // parameter for empirical bystander updates
    vph_type memb;   // phenotypes of members of the group
    unsigned numby;  // number of bystanders
    bool lhist;      // whether to collect learning history
    bool empi;       // whether to compute empirical scores
    vs_type stat;    // learning statistics
};

template<typename PhenType>
ActCritGroup<PhenType>::ActCritGroup(unsigned a_gs,
    unsigned a_spn,
    unsigned a_gnum,
    unsigned a_T,
    unsigned a_tau,
    flt a_V0,
    flt a_V,
    flt a_C,
    flt a_sigma,
    flt a_sigp,
    flt a_mf1,
    flt a_mf2,
    flt a_pmax,
    flt a_alphahat,
    flt a_betahat,
    const vph_type& a_memb,
    unsigned a_numby,
    bool a_lhist,
    bool a_empi) :
    gs{a_gs},
    spn{a_spn},
    gnum{a_gnum},
    T{a_T},
    tau{a_tau},
    V0{a_V0},
    V{a_V},
    C{a_C},
    sigma{a_sigma},
    sigp{a_sigp},
    mf1{a_mf1},
    mf2{a_mf2},
    pmax{a_pmax},
    alphahat{a_alphahat},
    betahat{a_betahat},
    memb{a_memb},
    numby{a_numby},
    lhist{a_lhist},
    empi{a_empi}
{
    if (lhist) {
        empi = true;
        stat.reserve(T);
    }
}

template<typename PhenType>
void ActCritGroup<PhenType>::Interact(rand_eng& eng)
{
    rand_uni uni(0, 1);
    rand_int uri(0, gs - 1);
    rand_norm eps(0, sigma);
    rand_norm eqd(0, sigp);
    // set payoff values to zero at start of generation
    for (auto& m : memb) {
        m.payoff = 0;
    }
    // run through the time steps; there should be on average 2*T/g rounds of
    // dominance interaction per group member
    for (unsigned tstep = 0; tstep < T; ++tstep) {
        // select random pair to interact
        unsigned i = 0;
        unsigned j = 1;
        // the above values for i and j are OK if gs == 2
        if (gs > 2) {
            i = uri(eng);
            j = uri(eng);
            while (j == i) j = uri(eng);
        }
        // apply forgetting
        if (mf1*mf2 < 1) {
            Forget(i, j, tstep);
            Forget(j, i, tstep);
        }

        phen_type& mi = memb[i];
        phen_type& mj = memb[j];
        // observations of relative quality
        flt xi_ij = mi.y - mj.z + eps(eng);
        flt xi_ji = mj.y - mi.z + eps(eng);
        // logit of probability using A
        flt lij = mi.l(xi_ij, j);
        flt lji = mj.l(xi_ji, i);
        // probability using A
        flt pij = 1/(1 + std::exp(-lij));
        flt pji = 1/(1 + std::exp(-lji));
        unsigned ui = (uni(eng) < Clamp(pij)) ? 1:0;
        unsigned uj = (uni(eng) < Clamp(pji)) ? 1:0;
        mi.nInts += 1;
        mj.nInts += 1;
        if (ui*uj > 0) {
            mi.nAA += 1;
            mj.nAA += 1;
        }
        flt Rij = R(i, j, ui, uj, eqd(eng));
        flt Rji = R(j, i, uj, ui, eqd(eng));
        // perform payoff increments
        flt val = (tstep >= tau) ? V : 0;
        Update_payoffs(i, j, ui, uj, val);
        // update actor-critic learning parameters
        flt deltaij = Rij - mi.vhat(xi_ij, j);
        flt deltaji = Rji - mj.vhat(xi_ji, i);
        if (empi) {
            // update lhat scores
            flt phatij = 1/(1 + std::exp(-mi.lp.lhat[j]));
            mi.lp.lhat[j] +=
                (ui == 1) ? (1 - phatij)*alphahat : -phatij*alphahat;
            flt phatji = 1/(1 + std::exp(-mj.lp.lhat[i]));
            mj.lp.lhat[i] +=
                (uj == 1) ? (1 - phatji)*alphahat : -phatji*alphahat;
            // update Elo scores
            flt pEloij = 1/(1 + std::exp(-(mi.EloR - mj.EloR)));
            if (ui == uj) {
                mi.EloR -= betahat*(pEloij - 0.5);
                mj.EloR += betahat*(pEloij - 0.5);
            } else if (ui == 1) {
                mi.EloR += betahat*(1 - pEloij);
                mj.EloR -= betahat*(1 - pEloij);
            } else { // uj == 1
                mi.EloR -= betahat*pEloij;
                mj.EloR += betahat*pEloij;
            }
        }
        if (lhist)  {
            Add_lstat(tstep, i, j, ui, uj, xi_ij, xi_ji, lij, lji, pij, pji,
                      Rij, Rji, deltaij, deltaji);
        }
        Update_w(i, j, xi_ij, deltaij);
        Update_w(j, i, xi_ji, deltaji);
        flt derij = (ui == 1) ? 1 - pij : -pij;
        flt derji = (uj == 1) ? 1 - pji : -pji;
        Update_theta(i, j, xi_ij, derij, deltaij);
        Update_theta(j, i, xi_ji, derji, deltaji);
        // perform bystander updates
        if (ui != uj && numby > 0) {
            // only update bystander when interaction is A-S or S-A
            // find numby distinct bystanders
            std::vector<unsigned> byst(gs - 2);
            unsigned l = 0;
            for (unsigned k = 0; k < gs; ++k) {
                if (k != i && k != j) {
                    byst[l++] = k;
                }
            }
            std::shuffle(byst.begin(), byst.end(), eng);
            // use the first numby individuals in shuffled vector
            for (unsigned m = 0; m < numby; ++m) {
                unsigned k = byst[m];
                phen_type& mk = memb[k];
                // very simple updating: add or subtract beta from th depending
                // on if focal individual (i or j) won or lost; note that
                // effects of generalisation on lpk.th[k] cancel out
                lp_type& lpk = mk.lp;
                flt Pij = 1/(1 + std::exp(lpk.th[i] - lpk.th[j]));
                if (ui == 1) {
                    lpk.th[i] += -(1 - Pij)*mk.beta;
                    lpk.th[j] += (1 - Pij)*mk.beta;
                } else {
                    lpk.th[i] += Pij*mk.beta;
                    lpk.th[j] += -Pij*mk.beta;
                }
                if (empi) {
                    flt Ptildeij = 1/(1 + std::exp(lpk.lhat[i] - lpk.lhat[j]));
                    if (ui == 1) {
                        lpk.lhat[i] += -(1 - Ptildeij)*mk.beta;
                        lpk.lhat[j] += (1 - Ptildeij)*mk.beta;
                    } else {
                        lpk.lhat[i] += Ptildeij*mk.beta;
                        lpk.lhat[j] += -Ptildeij*mk.beta;
                    }
                }
            }
        }
    }
    // scale payoff to be per interaction
    for (auto& m : memb) {
        if (m.nInts > 0) {
            m.payoff /= m.nInts;
        }
    }
}

template<typename PhenType>
void ActCritGroup<PhenType>::Forget(unsigned i, unsigned j, unsigned tstep)
{
    // move individual i towards 'naive' state with respect to j (the member
    // function presupposes that j != i)
    phen_type& mi = memb[i];
    lp_type& lpi = mi.lp;
    // deviation of 'generalised value' of w and th from initial value are
    // multiplied by memory factor 1 for each time step since last
    unsigned tdiffi = tstep - lpi.tlast[i];
    flt mf1i = std::pow(mf1, tdiffi);
    lpi.w[i] = mi.w0 + mf1i*(lpi.w[i] - mi.w0);
    // lpi.g = mi.g0 + mf1i*(lpi.g - mi.g0);
    lpi.th[i] = mi.th0 + mf1i*(lpi.th[i] - mi.th0);
    // lpi.ga = mi.ga0 + mf1i*(lpi.ga - mi.ga0);
    lpi.tlast[i] = tstep; // reset tlast component
    // deviations of components of w and th for j
    // from initial value for are multiplied by memory factor 2
    unsigned tdiffj = tstep - lpi.tlast[j];
    flt mf2j = std::pow(mf2, tdiffj);
    lpi.w[j] = mi.w0 + mf2j*(lpi.w[j] - mi.w0);
    lpi.th[j] = mi.th0 + mf2j*(lpi.th[j] - mi.th0);
    lpi.tlast[j] = tstep; // reset tlast component
}

template<typename PhenType>
typename ActCritGroup<PhenType>::flt
ActCritGroup<PhenType>::R(unsigned i, unsigned j,
                          unsigned ui, unsigned uj, flt eq)
{
    phen_type& mi = memb[i];
    phen_type& mj = memb[j];
    if (ui == 1) {
        if (uj == 1) {
            return mi.v - ce(mi.q, mj.q, eq);
        } else {
            return mi.v;
        }
    } else {
        return 0;
    }
}

template<typename PhenType>
void ActCritGroup<PhenType>::Update_payoffs(unsigned i, unsigned j,
                                            unsigned ui, unsigned uj, flt val)
{
    phen_type& mi = memb[i];
    phen_type& mj = memb[j];
    mi.payoff += V0;
    mj.payoff += V0;
    if (ui == 1) {
        if (uj == 1) {
            mi.payoff -= ce(mi.q, mj.q, 0);
            mj.payoff -= ce(mj.q, mi.q, 0);
        } else {
            mi.payoff += val;
        }
    } else {
        if (uj == 1) {
            mj.payoff += val;
        } else {
            mi.payoff += val/2;
            mj.payoff += val/2;
        }
    }
}

template<typename PhenType>
void ActCritGroup<PhenType>::Update_w(unsigned i, unsigned j,
                                      flt xi_ij, flt deltaij)
{
    phen_type& mi = memb[i];
    lp_type& lpi = mi.lp;
    flt alphwdij = mi.alphw*deltaij;
    // lpi.g += mi.gf*alphwdij*xi_ij;
    lpi.w[i] += mi.gf*alphwdij;
    lpi.w[j] += (1 - mi.gf)*alphwdij;
}

template<typename PhenType>
void ActCritGroup<PhenType>::Update_theta(unsigned i, unsigned j,
                                          flt xi_ij, flt derij, flt deltaij)
{
    phen_type& mi = memb[i];
    lp_type& lpi = mi.lp;
    flt alphthderdij = mi.alphth*derij*deltaij;
    // lpi.ga += mi.gf*alphthderdij*xi_ij;
    lpi.th[i] += mi.gf*alphthderdij;
    lpi.th[j] += (1 - mi.gf)*alphthderdij;
}

template<typename PhenType>
void ActCritGroup<PhenType>::Add_lstat(unsigned tstep,
                                       unsigned i, unsigned j,
                                       unsigned ui, unsigned uj,
                                       flt xi_ij, flt xi_ji,
                                       flt lij, flt lji,
                                       flt pij, flt pji,
                                       flt Rij, flt Rji,
                                       flt deltaij, flt deltaji)
{
    // for ease of interpretation, make sure i < j
    if (i > j) {
        unsigned k = i;
        i = j;
        j = k;
        unsigned uk = ui;
        ui = uj;
        uj = uk;
        flt xi = xi_ij;
        xi_ij = xi_ji;
        xi_ji = xi;
        flt p = pij;
        pij = pji;
        pji = p;
        flt l = lij;
        lij = lji;
        lji = l;
        flt R = Rij;
        Rij = Rji;
        Rji = R;
        flt delta = deltaij;
        deltaij = deltaji;
        deltaji = delta;
    }
    phen_type& mi = memb[i];
    phen_type& mj = memb[j];
    stat_type st;
    st.spn = spn;           // subpopulation number
    st.gnum = gnum;         // group number
    st.tstep = tstep;       // time step (round) for group
    st.i = i;
    st.j = j;
    st.ui = ui;
    st.uj = uj;
    st.lij = lij;
    st.lji = lji;
    st.pij = pij;
    st.pji = pji;
    st.Rij = Rij;
    st.Rji = Rji;
    st.dij = deltaij;
    st.dji = deltaji;
    st.xi_ij = xi_ij;
    st.xi_ji = xi_ji;
    st.gi = mi.lp.g;
    st.wii = mi.lp.w[i];
    st.wij = mi.lp.w[j];
    st.gj = mj.lp.g;
    st.wjj = mj.lp.w[i];
    st.wji = mj.lp.w[j];
    st.gai = mi.lp.ga;
    st.thii = mi.lp.th[i];
    st.thij = mi.lp.th[j];
    st.gaj = mj.lp.ga;
    st.thjj = mj.lp.th[j];
    st.thji = mj.lp.th[i];
    st.lhatij = mi.lp.lhat[j];
    st.lhatji = mj.lp.lhat[i];
    st.EloRi = mi.EloR;
    st.EloRj = mj.EloR;
    stat.push_back(st);
}

#endif // ACGROUP_HPP
