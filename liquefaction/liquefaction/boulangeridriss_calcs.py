# functions needed to run liquefaction calcs, based on Boulanger & Idriss (2014)
# Emily Mongold, 2022
#

from .base import *


def find_CSR75(d, M, pga, sig_v, sig_prime_v, q_c1Ncs):

    alpha = -1.012 - 1.126 * np.sin((d / 11.73) + 5.133)
    beta = 0.106 + 0.118 * np.sin((d / 11.28) + 5.142)
    r_d = np.exp(alpha + beta * M)
    CSR = 0.65 * (sig_v / sig_prime_v) * pga * r_d
    MSF = 6.9 * np.exp(- M / 4) - 0.058
    #     MSF_max = 1.09 + (q_c1Ncs / 180) ** 3
    #     MSF_max = np.array(list(map(lambda x:min(x,2.2),MSF_max)))
    #     MSF = 1 + (MSF_max - 1) * (8.64 * np.exp(-M / 4) - 1.325)
    Pa = 0.101325  # MPa
    C_sig = np.minimum(1 / (37.3 - 8.27 * q_c1Ncs ** 0.264),0.3)
    K_sig = np.minimum(1 - C_sig * np.log(sig_prime_v / Pa), 1.1)

    CSR_75 = CSR / (MSF * K_sig)

    return CSR_75


def find_CSR(d, M, pga, sig_v, sig_prime_v):

    alpha = -1.012 - 1.126 * np.sin((d / 11.73) + 5.133)
    beta = 0.106 + 0.118 * np.sin((d / 11.28) + 5.142)
    r_d = np.exp(alpha + beta * M)
    CSR = 0.65 * (sig_v / sig_prime_v) * pga * r_d

    return CSR


def find_CRR(M, sig_prime_v, q_c1Ncs):

    MSF = 6.9 * np.exp(- M / 4) - 0.058
    Pa = 0.101325  # MPa
    C_sig = np.minimum(1 / (37.3 - 8.27 * q_c1Ncs ** 0.264),0.3)
    K_sig = np.minimum(1 - C_sig * np.log(sig_prime_v / Pa), 1.1)

    CRR_75 = np.exp(q_c1Ncs / 113 + (q_c1Ncs / 1000) ** 2 - (q_c1Ncs / 140) ** 3 + (q_c1Ncs / 137) ** 4 - 2.8)
    CRR = CRR_75 * MSF * K_sig

    return CRR


def iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol, C_FC=0):
    Pa = 0.101325  # MPa
    m0 = 1.338 - 0.249 * q_c ** 0.264  # initialize m for calculation of C_N

    C_N = np.minimum((Pa / sig_prime_v) ** m0, 1.7)
    # C_N = np.array(list(map(lambda x: min(x, 1.7), C_N)))
    q_c1N = C_N * q_c / Pa

    I_c = solve_Ic(q_c, sig_v, sig_prime_v, f_s)
    # C_FC = 0  # This is a central value, can be positive or negative-- 0.07 is good for the liquefiable soils in Christchurch, NZ
    FC = 80 * (I_c + C_FC) - 137

    dq_c1N = (11.9 + q_c1N / 14.6) * np.exp(1.63 - 9.7 / (FC + 2) - (15.7 / (FC + 2)) ** 2)

    q_c1Ncs = q_c1N + dq_c1N

    # m = 1.338 - 0.249 * q_c1Ncs ** 0.264

    count = 0
    while any(abs(m0 - (1.338 - 0.249 * q_c1Ncs ** 0.264)) > tol) and count <= num:
        m0 = 1.338 - 0.249 * q_c1Ncs ** 0.264
        C_N = np.minimum((Pa / sig_prime_v) ** m0,1.7)
        # C_N = np.array(list(map(lambda x: min(x, 1.7), C_N)))
        q_c1N = C_N * q_c / Pa
        dq_c1N = (11.9 + q_c1N / 14.6) * np.exp(1.63 - 9.7 / (FC + 2) - (15.7 / (FC + 2)) ** 2)
        q_c1Ncs = q_c1N + dq_c1N
        count += 1

    return q_c1Ncs


def bi_prob(cpt, mags, pgas, C_FC=0, sim=None):
    # Constants outside of loops
    sig_lnR = 0.2  # recommended for deterministic
    num = 50
    tol = 10 ** -7
    probs = {}
    for key in cpt:
        try:
            probs[key] = np.array(len(mags))
            temp = np.zeros(len(mags))
        except:
            probs[key] = np.array(1)
            temp = np.zeros(1)

        d = np.array(cpt[key]['CPT_data']['d'])
        q_c = np.array(cpt[key]['CPT_data']['q_c'])
        f_s = np.array(cpt[key]['CPT_data']['f_s'])
        R_f = np.array(cpt[key]['CPT_data']['R_f'])
        sig_v = np.array(cpt[key]['CPT_data']['sig_v'])
        sig_prime_v = np.array(list(map(lambda x: x[0], cpt[key]['sig_prime_v'])))

        if sim is not None:
            M = mags
            pga = pgas[key][sim]

            q_c1Ncs = iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol, C_FC)
            CSR_75 = find_CSR75(d, M, pga, sig_v, sig_prime_v, q_c1Ncs)
            #     sig_lnR = -0.2/norm.ppf(0.16) # deterministic solution
            P_L = norm.cdf((-q_c1Ncs / 113 - (q_c1Ncs / 1000) ** 2 + (q_c1Ncs / 140) ** 3 - (
                    q_c1Ncs / 137) ** 4 + 2.6 + np.log(CSR_75)) / sig_lnR)
            temp = max(P_L)

        else:
            for scen, (M,pga) in enumerate(zip(mags,pgas[key])):
                q_c1Ncs = iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol, C_FC)
                CSR_75 = find_CSR75(d, M, pga, sig_v, sig_prime_v, q_c1Ncs)
                #     sig_lnR = -0.2/norm.ppf(0.16) # deterministic solution
                P_L = norm.cdf((-q_c1Ncs / 113 - (q_c1Ncs / 1000) ** 2 + (q_c1Ncs / 140) ** 3 - (
                            q_c1Ncs / 137) ** 4 + 2.6 + np.log(CSR_75)) / sig_lnR)
                temp[scen] = np.nanmax(P_L)
                # temp[scen] = max(P_L)

        probs[key] = temp
    return probs


def bi_lpi(cpt, mags, pgas, C_FC=0, sim=None):
    # apply the FS/LPI method to CSR and CRR values estimated from calculations from Boulanger and Idriss (2014)
    # Input cpt dictionary with CPT data
    # Input mags array of magnitudes
    # Input pgas dictionary with PGA values
    # Input C_FC central value for fines content constant
    # Input sim simulation number
    # Output LPIs dictionary with LPI values
    num = 50
    tol = 10 ** -7
    LPIs = {}
    for key in cpt:
        d = np.array(cpt[key]['CPT_data']['d'])
        q_c = np.array(cpt[key]['CPT_data']['q_c'])
        f_s = np.array(cpt[key]['CPT_data']['f_s'])
        sig_v = np.array(cpt[key]['CPT_data']['sig_v'])
        sig_prime_v = np.array(list(map(lambda x: x[0], cpt[key]['sig_prime_v'])))
        I_c = solve_Ic(q_c, sig_v, sig_prime_v, f_s)
        nonliq = np.where(I_c > 2.6)

        if sim is not None:
            M = mags
            pga = pgas[key][sim]
            q_c1Ncs = iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol, C_FC)
            CSR = find_CSR(d, M, pga, sig_v, sig_prime_v)
            CRR = find_CRR(M, sig_prime_v, q_c1Ncs)
            FS = np.clip(CRR / CSR, 0, 1)
            FS[nonliq] = 1
            w = 10 - 0.5 * d
            lpi = sum((1 - FS) * w * cpt[key]['CPT_data']['dz'])

        else:
            lpi = np.zeros(len(mags))
            q_c1Ncs = iterate_q(q_c, sig_prime_v, sig_v, f_s, num, tol)
            w = 10 - 0.5*d
            for scen, (M,pga) in enumerate(zip(mags,pgas[key])):
                CSR = find_CSR(d, M, pga, sig_v, sig_prime_v)
                CRR = find_CRR(M, sig_prime_v, q_c1Ncs)
                FS = np.clip(CRR/CSR,0,1)
                FS[nonliq] = 1
                lpi[scen] = sum((1-FS)*w*cpt[key]['CPT_data']['dz'])

        LPIs[key] = lpi
    return LPIs


def solve_Ic(q_c, sig_v, sig_prime_v, f_s):
    # solves for the soil behavior index, Ic using an interative n procedure as in Robertson (2009)
    # input q_c tip resistance
    # input sig_v vertical soil stress
    # input sig_prime_v effective vertical soil stress
    # output Ic soil behavior type index
    Pa = 0.101325
    n = np.full_like(sig_prime_v,1.0)  # initialize n array with same shape as sig_prime_v
    tol = 1e-3
    # if len(sig_prime_v.shape) > 1:
    #     sig_prime_v = np.array(list(map(lambda x: x[0], sig_prime_v)))

    Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n
    F = 100 * f_s / (q_c - sig_v)
    Q = np.maximum(1, Q)
    F = np.where(F < 0, np.maximum(1, F), F)
    Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)

    n2 = 0.381*Ic + 0.05 * (sig_prime_v / Pa) - 0.15
    n2 = np.maximum(n2, 0.5)
    n2 = np.minimum(n2, 1.0)
    while any(abs(n-n2) > tol):
        Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n2
        Q = np.maximum(1, Q)
        Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
        n = n2
        n2 = 0.381 * Ic + 0.05 * (sig_prime_v / Pa) - 0.15
        n2 = np.maximum(n2, 0.5)
        n2 = np.minimum(n2, 1.0)

    return Ic
