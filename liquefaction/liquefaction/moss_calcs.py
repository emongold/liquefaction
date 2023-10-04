# functions needed to run liquefaction calcs, based on Moss et al. (2006)
# Emily Mongold, 2022
#

from .base import *
from .preprocess import interp_gwt


def soil_stress(cpt, slr):
    # return the sigma and sigma v based on cpt data
    # input: cpt dict with data for each borehole
    # input: slr (can be an array)
    # output: cpt dict with additional data for soil stress
    # Constants
    g = 9.81
    # Pa = 0.101325  # MPa = 1.033kg/cm2, = 14.696 psi = 1.058ft^2
    rho_w = 1  # density of water in Mg/m^3
    gamma_w = rho_w * g / 1000  # in units of MPa
    table = {}
    if len(cpt) > 3:
        cpt = interp_gwt(cpt)
    for key,data in cpt.items():
        table[key] = {}
        table[key]['emergent_flag'] = np.zeros(len(slr))
        # R_f = np.zeros(len(data['CPT_data']))
        u = np.zeros(shape=(len(data['CPT_data']), len(slr)))

        for rise in range(len(slr)):
            if slr[rise] > data['Water depth']:
                table[key]['emergent_flag'][rise] = 1
                h = data['CPT_data']['d']
            else:
                h = np.maximum(data['CPT_data']['d'] - data['Water depth'] + slr[rise], 0)
                # h = np.array(list(map(lambda x: max(x, 0), h)))
            for i in range(len(u)):
                u[:, rise] = gamma_w * h

        sig_v = np.zeros(len(data['CPT_data']))
        for i in range(len(data['CPT_data'])):
            # sig_v[i] = sig_v[i - 1] + data['CPT_data']['dsig_v'][i] # This only works when data is ordered by d
            d = data['CPT_data']['d'][i]
            sig_v[i] = sum(data['CPT_data']['dsig_v'][data['CPT_data']['d'] <= d])
        # sig_prime_v = sig_v[:,np.newaxis] - u

        sig_prime_v = u
        for i in range(u.shape[1]):
            sig_prime_v[:,i] = sig_v - u[:,i]

        data['CPT_data']['sig_v'] = sig_v
        data['sig_prime_v'] = sig_prime_v

        table[key]['wd'] = np.maximum(np.round(data['Water depth'] - np.array(slr),1), 0).tolist()
        # table[key]['wd'] = list(map(lambda x: max(round(x, 1), 0), table[key]['wd']))

    return cpt, table

def solve_FS(cpt, mags, pgas, slr, mc=None):
    # function to determine the factor of safety against liquefaction over the depth of each borehole
    # input: cpt dict with each borehole and soil data over the depth
    # input: mags vector of magnitudes for each scenario
    # input: pgas matrix of pgas for each simulation of each scenario
    # output: cpt updated dict with additional FS and CRR/CSR values
    # output: depth dict with dataframe of d and dz for each borehole

    # Constants
    # x1 = 0.78
    # x2 = -0.33
    # y1 = -0.32
    # y2 = -0.35
    # y3 = 0.49
    # z1 = 1.21
    Pa = 0.101325
    g = 9.81

    FS = {}
    depth = {}

    if mc is not None:
        for rise in range(len(slr)):
            SLR = str(slr[rise])
            FS[SLR] = {}
            for bh in range(len(cpt)):
                key = list(cpt.keys())[bh]
                FS[SLR][key] = np.zeros(len(cpt[key]['CPT_data']))

        for bh in range(len(cpt)):
            key = list(cpt.keys())[bh]
            M = mags
            pga = pgas[key][mc]

            r_d = np.zeros(len(cpt[key]['CPT_data']))
            num = -9.147 - 4.173 * pga + 0.652 * M
            for z in range(len(cpt[key]['CPT_data'])):
                d = cpt[key]['CPT_data']['d'][z]
                den = 10.567 + 0.089 * np.exp(0.089 * (-(d * 3.28) - (7.76 * pga) + 78.576))
                rd_num = 1 + (num / den)
                den = 10.567 + 0.089 * np.exp(0.089 * (-(7.76 * pga) + 78.576))
                rd_den = 1 + (num / den)
                if d <= 20:
                    r_d[z] = rd_num / rd_den
                elif d > 20:
                    # this should never be the case
                    r_d[z] = rd_num / rd_den - 0.0014 * (d * 3.28 - 65)

            for rise in range(len(slr)):
                SLR = str(slr[rise])

                q_c1, c, C_q = iterate_c(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['R_f'],
                                         cpt[key]['sig_prime_v'][:, rise])

                # CSR = np.zeros(len(cpt[key]['CPT_data']))
                CRR = np.full((len(cpt[key]['CPT_data'])), np.nan)

                CSR = 0.65 * pga * (cpt[key]['CPT_data']['sig_v'] / cpt[key]['sig_prime_v'][:, rise]) * r_d
                CSR = list(map(lambda x: max(1e-5, x), CSR)) # eliminate negative CSR values
                I_c = solve_Ic(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['sig_v'],
                               cpt[key]['sig_prime_v'][:, rise], cpt[key]['CPT_data']['f_s'])

                for i in range(len(cpt[key]['CPT_data'])):
                    if I_c[i] > 2.6:
                        CRR[i] = 2 * CSR[i]  # no liquefaction
                    else:
                        try:
                            CRR[i] = np.exp(
                                (q_c1[i] ** 1.045 + q_c1[i] * (0.110 * cpt[key]['CPT_data']['R_f'][i]) +
                                 (0.001 * cpt[key]['CPT_data']['R_f'][i]) +
                                 c[i] * (1 + 0.85 * cpt[key]['CPT_data']['R_f'][i]) -
                                 0.848 * np.log(M) - 0.002 * np.log(cpt[key]['sig_prime_v'][i, rise]) - 20.923 +
                                 1.632 * norm.ppf(0.15)) / 7.177)
                        except:
                            CRR[i] = CSR[i] * 2
                FS[SLR][key] = CRR / CSR
            d = np.where(cpt[key]['CPT_data'].keys() == 'd')[0][0]
            dz = np.where(cpt[key]['CPT_data'].keys() == 'dz')[0][0]
            depth[key] = cpt[key]['CPT_data'].iloc[:, [d, dz]]

    else:

        for rise in range(len(slr)):
            SLR = str(slr[rise])
            FS[SLR] = {}
            for bh in range(len(cpt)):
                key = list(cpt.keys())[bh]
                FS[SLR][key] = np.zeros(
                    shape=(np.shape(pgas[key])[0], len(cpt[key]['CPT_data']), np.shape(pgas[key])[1]))

        for bh in range(len(cpt)):
            key = list(cpt.keys())[bh]

            for scen in range(len(pgas[key])):

                M = mags[scen]
                DWF_M = 17.84 * (M ** (-1.43))
                pga = pgas[key][scen]  # length is nsim

                r_d = np.zeros(shape=(len(cpt[key]['CPT_data']), len(pga)))
                num = -9.147 - 4.173 * pga + 0.652 * M
                for z in range(len(cpt[key]['CPT_data'])):
                    d = cpt[key]['CPT_data']['d'][z]
                    den = 10.567 + 0.089 * np.exp(0.089 * (-(d * 3.28) - (7.76 * pga) + 78.576))
                    rd_num = 1 + (num / den)
                    den = 10.567 + 0.089 * np.exp(0.089 * (-(7.76 * pga) + 78.576))
                    rd_den = 1 + (num / den)
                    if d <= 20:
                        r_d[z] = rd_num / rd_den
                    elif d > 20:
                        # this should never be the case
                        r_d[z] = rd_num / rd_den - 0.0014 * (d * 3.28 - 65)

                for rise in range(len(slr)):
                    SLR = str(slr[rise])

                    q_c1, c, C_q = iterate_c(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['R_f'],
                                             cpt[key]['sig_prime_v'][:, rise])

                    CSR = np.zeros(shape=(len(cpt[key]['CPT_data']), len(pga)))
                    #     temp = np.zeros(shape=(len(cpt[key]['CPT_data']),len(pga)))
                    CRR = np.full((len(cpt[key]['CPT_data']), len(pga)), np.nan)

                    for sim in range(len(pga)):
                        # do not divide a_max by g since pga is already in units of g:
                        CSR[:, sim] = 0.65 * (pga[sim]) * (
                                cpt[key]['CPT_data']['sig_v'] / cpt[key]['sig_prime_v'][:, rise]) * r_d[:, sim]
                        # CSR = list(map(lambda x: max(1e-5, x), CSR)) # eliminate negative CSR values

                        # Q = ((cpt[key]['CPT_data']['q_c'] - cpt[key]['CPT_data']['sig_v']) / Pa) * (
                        #             Pa / cpt[key]['sig_prime_v'][:, rise]) ** n
                        # F = 100 * cpt[key]['CPT_data']['f_s'] / (
                        #             cpt[key]['CPT_data']['q_c'] - cpt[key]['CPT_data']['sig_v'])
                        # I_c = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
                        I_c = solve_Ic(cpt[key]['CPT_data']['q_c'], cpt[key]['CPT_data']['sig_v'],
                                       cpt[key]['sig_prime_v'][:, rise], cpt[key]['CPT_data']['f_s'])

                        for i in range(len(cpt[key]['CPT_data'])):
                            # temp[i,sim] = q_c1[i]**1.045 + q_c1[i]*(0.110*R_f[i]) + (0.001*R_f[i]) + c[i]*(1+0.85*R_f[i])
                            #  - 7.177*np.log(CSR[i,sim]) - 0.848*np.log(M) - 0.002*np.log(sig_prime_v[i,sim]) - 20.923

                            if (I_c[i] > 2.6):
                                CRR[i, sim] = 2 * CSR[i, sim]  # to try to match Cliq
                            else:
                                try:
                                    CRR[i, sim] = np.exp(
                                        (q_c1[i] ** 1.045 + q_c1[i] * (0.110 * cpt[key]['CPT_data']['R_f'][i]) +
                                         (0.001 * cpt[key]['CPT_data']['R_f'][i]) +
                                         c[i] * (1 + 0.85 * cpt[key]['CPT_data']['R_f'][i]) -
                                         0.848 * np.log(M) - 0.002 * np.log(cpt[key]['sig_prime_v'][i, rise]) - 20.923 +
                                         1.632 * norm.ppf(0.15)) / 7.177)
                                except:
                                    CRR[i, sim] = CSR[i, sim] * 2
                        #     val = -temp/1.632 # not outputting this for now...
                        # cumulative normal of val
                        #     P_L all= np.array(norm.cdf(val))[:,0] #not outputting this for now...

                    # CSR = list(map(lambda x: max(1e-5, x), CSR)) # eliminate negative CSR values
                    FS[SLR][key][scen, :, :] = CRR / CSR

            d = np.where(cpt[key]['CPT_data'].keys() == 'd')[0][0]
            dz = np.where(cpt[key]['CPT_data'].keys() == 'dz')[0][0]
            depth[key] = cpt[key]['CPT_data'].iloc[:, [d, dz]]

    return depth, FS

def solve_LPI(depth, FS, table):
    # function to determine the liquefaction potential index given factor of safety values
    # input: depth dict with each borehole depth and dz values
    # input: FS dict of factors of safety against liquefaction for each slr scenario, borehole, scenario and simulation
    # output: LPIs dict with liquefaction potential index for each slr scenario, borehole, scenario and simulation

    names = list(FS[list(FS.keys())[0]].keys())
    LPIs = {}
    for rise in range(len(FS)):
        SLR = list(FS.keys())[rise]
        LPIs[SLR] = {}

        for bh in range(len(names)):
            key = names[bh]
            if len(FS[SLR][key].shape) == 1:
                ndepth = len(FS[SLR][key])
                F = np.zeros(ndepth)
                for k in range(ndepth):
                    if depth[key]['d'][k] < table[key]['wd'][rise]:
                        F[k] = 0
                    elif FS[SLR][key][k] < 0:  ## added in case FS is negative then F will not be >> 1
                        F[k] = 1
                    elif FS[SLR][key][k] < 1:
                        F[k] = 1 - FS[SLR][key][k]
                tempLPI = F * depth[key]['dz'] * (10 - 0.5 * depth[key]['d'])
                LPIs[SLR][key] = sum(tempLPI)

            else:
                LPIs[SLR][key] = np.zeros(shape=(np.shape(FS[SLR][key])[0], np.shape(FS[SLR][key])[2]))

                nscen = np.shape(FS[SLR][key])[0]
                ndepth = np.shape(FS[SLR][key])[1]
                nsim = np.shape(FS[SLR][key])[2]

                for scen in range(nscen):
                    for sim in range(nsim):
                        F = np.zeros(ndepth)
                        for k in range(ndepth):
                            if depth[key]['d'][k] < table[key]['wd'][rise]:
                                F[k] = 0
                            elif FS[SLR][key][scen, k, sim] < 0:  ## added in case FS is negative then F will not be >> 1
                                F[k] = 1
                            elif FS[SLR][key][scen, k, sim] < 1:
                                F[k] = 1 - FS[SLR][key][scen, k, sim]
                        tempLPI = F * depth[key]['dz'] * (10 - 0.5 * depth[key]['d'])
                        LPIs[SLR][key][scen, sim] = sum(tempLPI)
                for i in LPIs:
                    for j in LPIs[i]:
                        rep = []
                        for k in range(len(LPIs[i][j])):
                            rep.append(LPIs[i][j][k][0])
                        LPIs[i][j] = rep

    return LPIs

def agg(cpt):
    # function to aggregate cpt data every meter
    # input: cpt dict with dataframes for cpt data at each borehole
    # output: cpt new dict with dataframes with aggregate cpt data at each meter in the borehole- averaged from the
    #       original

    g = 9.81
    Pa = 0.101325  # MPa
    rho_w = 1000  # kg/m^3
    gamma_w = rho_w * g / 1000  # kPa
    for key in list(cpt.keys()):
        test = cpt[key]['CPT_data']
        test['meter'] = np.floor(test['d'])
        test = test.groupby('meter').mean()[['q_c', 'f_s', 'd']]
        test['R_f'] = 100 * test['f_s']/test['q_c']
        test['dz'] = 1
        gamma = np.zeros(len(test))

        gamma[0] = gamma_w * (0.27 * (np.log10(test['R_f'][0])) + 0.36 * np.log10(test['q_c'][0] / Pa) + 1.236)
        for i in range(1, len(test)):
            # Calculating soil unit weight from Robertson and Cabal (2010)
            if test['R_f'][i] == 0:
                if test['q_c'][i] == 0:
                    gamma[i] = gamma_w * 1.236
                else:
                    test['gamma'][i] = gamma_w * (0.36 * np.log10(test['q_c'][i] / Pa) + 1.236)
            elif test['q_c'][i] == 0:
                gamma[i] = gamma_w * (0.27 * (np.log10(test['R_f'][i])) + 1.236)
            else:
                gamma[i] = gamma_w * (0.27 * (np.log10(test['R_f'][i])) + 0.36 * np.log10(test['q_c'][i] / Pa) + 1.236)

        test['gamma'] = gamma
        test['dsig_v'] = test['dz'] * test['gamma']
        cpt[key]['CPT_data'] = test
        del test
    return cpt

def iterate_c(q_c, R_f, sig_pv):
    # constants
    x1 = 0.78
    x2 = -0.33
    y1 = -0.32
    y2 = -0.35
    y3 = 0.49
    z1 = 1.21
    Pa = 0.101325
    tol = 1e-6

    c = np.zeros(len(R_f))
    for k in range(len(c)):
        f1 = x1 * q_c[k] ** x2
        f2 = -(y1 * (q_c[k] ** y2) + y3)
        f3 = abs(np.log10(10 + q_c[k])) ** z1
        c[k] = f1 * (R_f[k] / f3) ** f2

    C_q = (Pa / sig_pv) ** c
    # C_q = np.array(list(map(lambda x: min(x, 1.7), C_q.diagonal())))
    C_q = np.array(list(map(lambda x: min(x, 1.7), C_q)))
    q_c1 = C_q * q_c

    c_new = np.zeros(len(c))
    for k in range(len(c_new)):
        f1 = x1 * q_c1[k] ** x2
        f2 = -(y1 * (q_c1[k] ** y2) + y3)
        f3 = abs(np.log10(10 + q_c1[k])) ** z1
        c_new[k] = f1 * (R_f[k] / f3) ** f2

    while any(abs(c - c_new) > tol):
        C_q = (Pa / sig_pv) ** c_new
        # C_q = np.array(list(map(lambda x: min(x, 1.7), C_q.diagonal())))
        C_q = np.array(list(map(lambda x: min(x, 1.7), C_q)))
        q_c1 = C_q * q_c
        c = c_new
        for k in range(len(c_new)):
            f1 = x1 * q_c1[k] ** x2
            f2 = -(y1 * (q_c1[k] ** y2) + y3)
            f3 = abs(np.log10(10 + q_c1[k])) ** z1
            c_new[k] = f1 * (R_f[k] / f3) ** f2
    return q_c1, c_new, C_q

def solve_Ic(q_c, sig_v, sig_prime_v, f_s):
    # solves for the soil behavior index, Ic using an interative n procedure as in Robertson (2009)
    # input q_c tip resistance
    # input sig_v vertical soil stress
    # input sig_prime_v effective vertical soil stress
    # output Ic soil behavior type index
    Pa = 0.101325
    n = 1.0  # initialize
    tol = 1e-3
    if len(sig_prime_v.shape) > 1:
        sig_prime_v = np.array(list(map(lambda x: x[0], sig_prime_v)))

    Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n
    F = 100 * f_s / (q_c - sig_v)
    Q = np.array(list(map(lambda x: max(1, x), Q)))
    F = np.array(list(map(lambda x: max(1, x) if x < 0 else x, F)))
    Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)

    n2 = 0.381*Ic + 0.05 * (sig_prime_v / Pa) - 0.15
    n2 = np.array(list(map(lambda x: max(x, 0.5), n2)))
    n2 = np.array(list(map(lambda x: min(x, 1.0), n2)))
    while any(abs(n-n2) > tol):
        Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n2
        Q = np.array(list(map(lambda x: max(1, x), Q)))
        Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
        n = n2
        n2 = 0.381 * Ic + 0.05 * (sig_prime_v / Pa) - 0.15
        n2 = np.array(list(map(lambda x: max(x, 0.5), n2)))
        n2 = np.array(list(map(lambda x: min(x, 1.0), n2)))

    return Ic

