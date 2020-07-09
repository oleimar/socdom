# Run SocDomGen program and add some results to file,

source("h5_funcs.R")

reps <- 25

for (rep in 1:reps) {

system("../build/SocDomGen Run31b.toml")

d1 <- h5_dt("Run31b.h5")
av_w0 <- with(d1, mean(w0))
sd_w0 <- with(d1, sd(w0))
av_th0 <- with(d1, mean(th0))
sd_th0 <- with(d1, sd(th0))
av_g0 <- with(d1, mean(g0))
sd_g0 <- with(d1, sd(g0))
av_ga0 <- with(d1, mean(ga0))
sd_ga0 <- with(d1, sd(ga0))
av_alphw <- with(d1, mean(alphw))
sd_alphw <- with(d1, sd(alphw))
av_alphth <- with(d1, mean(alphth))
sd_alphth <- with(d1, sd(alphth))
av_beta <- with(d1, mean(beta))
sd_beta <- with(d1, sd(beta))
av_v <- with(d1, mean(v))
sd_v <- with(d1, sd(v))
av_gf <- with(d1, mean(gf))
sd_gf <- with(d1, sd(gf))
dr <- data.frame(av_w0 = av_w0,
                 av_th0 = av_th0,
                 av_g0 = av_g0,
                 av_ga0 = av_ga0,
                 av_alphw = av_alphw,
                 av_alphth = av_alphth,
                 av_beta = av_beta,
                 av_v = av_v,
                 av_gf = av_gf,
                 sd_w0 = sd_w0,
                 sd_th0 = sd_th0,
                 sd_g0 = sd_g0,
                 sd_ga0 = sd_ga0,
                 sd_alphw = sd_alphw,
                 sd_alphth = sd_alphth,
                 sd_beta = sd_beta,
                 sd_v = sd_v,
                 sd_gf = sd_gf
                 )

dr1 <- read.delim("Run31b_data.tsv")
dr1 <- rbind(dr1, dr)
write.table(dr1, "Run31b_data.tsv", quote = FALSE,
            sep = "\t", row.names = FALSE)

}
