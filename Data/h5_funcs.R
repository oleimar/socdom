# functions to read data from hdf5 file created by simulation program

# return data table for "one-column" phenotype data
h5_dt <- function(hf_name) {
    require(hdf5r)
    require(data.table)
    f.h5 <- H5File$new(hf_name, mode = "r")
    w0 <- f.h5[["w0"]][]
    th0 <- f.h5[["th0"]][]
    g0 <- f.h5[["g0"]][]
    ga0 <- f.h5[["ga0"]][]
    alphw <- f.h5[["alphw"]][]
    alphth <- f.h5[["alphth"]][]
    beta <- f.h5[["beta"]][]
    v <- f.h5[["v"]][]
    gf <- f.h5[["gf"]][]
    q <- f.h5[["q"]][]
    y <- f.h5[["y"]][]
    z <- f.h5[["z"]][]
    payoff <- f.h5[["payoff"]][]
    EloR <- f.h5[["EloR"]][]
    nInts <- f.h5[["nInts"]][]
    nAA <- f.h5[["nAA"]][]
    inum <- f.h5[["inum"]][] + 1
    gnum <- f.h5[["gnum"]][] + 1
    spn <- f.h5[["spn"]][] + 1
    female <- f.h5[["female"]][]
    alive <- f.h5[["alive"]][]
    g <- f.h5[["g"]][]
    ga <- f.h5[["ga"]][]
    f.h5$close_all()
    data.table(w0 = w0, th0 = th0, g0 = g0, ga0 = ga0,
               alphw = alphw, alphth = alphth, beta = beta,
               v = v, gf = gf, q = q, y = y, z = z,
               payoff = payoff, EloR = EloR,
               nInts = nInts, nAA = nAA, inum = inum, gnum = gnum,
               spn = spn, female = female, alive = alive,
               g = g, ga = ga)
}

# return matrix where each row is a learning parameter w
h5_w <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    w <- t(f.h5[["w"]][,])
    f.h5$close_all()
    w
}

# return matrix where each row is a learning parameter th
h5_th <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    th <- t(f.h5[["th"]][,])
    f.h5$close_all()
    th
}

# return matrix where each row is a parameter lhat
h5_lhat <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    lhat <- t(f.h5[["lhat"]][,])
    f.h5$close_all()
    lhat
}

# return matrix where each row is a maternal gamete value
h5_mat_gam <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    mat_gam <- t(f.h5[["MatGam"]][,])
    f.h5$close_all()
    mat_gam
}

# return matrix where each row is a paternal gamete value
h5_pat_gam <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    pat_gam <- t(f.h5[["PatGam"]][,])
    f.h5$close_all()
    pat_gam
}


# return data table for learning history data
h5_hdt <- function(hf_name) {
    require(hdf5r)
    require(data.table)
    f.h5 <- H5File$new(hf_name, mode = "r")
    spn <- f.h5[["spn"]][] + 1
    gnum <- f.h5[["gnum"]][] + 1
    tstep <- f.h5[["tstep"]][] + 1
    i <- f.h5[["i"]][] + 1
    j <- f.h5[["j"]][] + 1
    ui <- f.h5[["ui"]][]
    uj <- f.h5[["uj"]][]
    lij <- f.h5[["lij"]][]
    lji <- f.h5[["lji"]][]
    pij <- f.h5[["pij"]][]
    pji <- f.h5[["pji"]][]
    Rij <- f.h5[["Rij"]][]
    Rji <- f.h5[["Rji"]][]
    dij <- f.h5[["dij"]][]
    dji <- f.h5[["dji"]][]
    xi_ij <- f.h5[["xi_ij"]][]
    xi_ji <- f.h5[["xi_ji"]][]
    gi <- f.h5[["gi"]][]
    wii <- f.h5[["wii"]][]
    wij <- f.h5[["wij"]][]
    gj <- f.h5[["gj"]][]
    wjj <- f.h5[["wjj"]][]
    wji <- f.h5[["wji"]][]
    gai <- f.h5[["gai"]][]
    thii <- f.h5[["thii"]][]
    thij <- f.h5[["thij"]][]
    gaj <- f.h5[["gaj"]][]
    thjj <- f.h5[["thjj"]][]
    thji <- f.h5[["thji"]][]
    lhatij <- f.h5[["lhatij"]][]
    lhatji <- f.h5[["lhatji"]][]
    EloRi <- f.h5[["EloRi"]][]
    EloRj <- f.h5[["EloRj"]][]
    f.h5$close_all()
    data.table(spn = spn, gnum = gnum, tstep = tstep,
               i = i, j = j, ui = ui, uj = uj,
               lij = lij, lji = lji, pij = pij, pji = pji,
               Rij = Rij, Rji = Rji, dij = dij, dji = dji,
               xi_ij = xi_ij, xi_ji = xi_ji,
               gi = gi, wii = wii, wij = wij,
               gj = gj, wjj = wjj, wji = wji,
               gai = gai, thii = thii, thij = thij,
               gaj = gaj, thjj = thjj, thji = thji,
               lhatij = lhatij, lhatji = lhatji, EloRi = EloRi, EloRj = EloRj)
}

