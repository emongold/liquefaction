# script to plot outputs of the liquefaction postprocessing
# Emily Mongold, 2022

from .base import *
# import plotly.express as px
# import geoplot
import matplotlib as mpl
import matplotlib.pyplot as plt
# import numpy as np
# import json
# import geopandas as gpd


plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the y tick labels
plt.rc('legend', fontsize=12)  # fontsize of the legend
plt.rcParams["figure.figsize"] = (10, 7)
orig_cmap = plt.cm.GnBu
cols = orig_cmap(np.linspace(1, 0.3, 10))
cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", cols)


def Alameda_baseplot(plot_folder = 'alameda_plots/'):
    boundary_filename = plot_folder + 'Alameda_shape.geojson'
    buildings_filename = plot_folder + 'alameda_buildings.shp'
    boundary_shape = gpd.read_file(boundary_filename)
    building_shape = gpd.read_file(buildings_filename)

    plt.figure(figsize=(14, 9))
    ax = plt.subplot(111)
    geoplot.polyplot(boundary_shape, facecolor="None", edgecolor="black", ax=ax,
                     linewidth=1.0, extent=(-122.35, 37.745, -122.22, 37.80), zorder=3, alpha=1)
    geoplot.polyplot(building_shape, facecolor="grey", edgecolor="None", ax=ax,
                     linewidth=1.0, extent=(-122.35, 37.745, -122.22, 37.80), zorder=3, alpha=0.5)
    return ax

def meanLPI(LPIs):
    # input LPIs is the dict of all LPI values for each SLR scenario, each borehole, and each scen/sim
    # input SLR is the name of the scenario (string)
    # output meanLPIs is the expected LPI value for each scenario at each borehole location

    meanLPIs = {}
    for rise in range(len(LPIs)):
        SLR = list(LPIs.keys())[rise]
        meanLPIs[SLR] = {}
        for bh in range(len(LPIs[SLR])):
            key = list(LPIs[SLR].keys())[bh]
            meanLPIs[SLR][key] = np.mean(LPIs[SLR][key], 1)

    return meanLPIs

def restruct(LPIs, SLR, mags, dist):
    # take LPIs and make single dataframe with rows for bh/slr info
    dflpi = pd.DataFrame()
    for i in range(len(SLR)):
        slr = SLR[i]
        for bh in range(len(LPIs[slr])):
            key = list(LPIs[slr].keys())[bh]
            temp = pd.DataFrame(LPIs[slr][key])
            temp['avg'] = np.mean(pd.DataFrame(LPIs[slr][key]), 1)
            temp['bh'] = key
            temp['slr'] = slr
            temp['mags'] = mags
            temp['dist'] = dist
            dflpi = pd.concat([dflpi, temp], ignore_index=True)

    dflpi.reset_index(inplace=True, drop=True)

    return dflpi

def get_dists(imdir):
    # function to get a list of the magnitudes of the events from the output of R2D
    # input: imdir is the directory to the output R2D folder
    # output: mags is a list of the magnitudes of each scenario
    with open(imdir + 'SiteIM.json', 'r') as j:
        contents = json.loads(j.read())

    dist = []
    for scen in range(len(contents['Earthquake_MAF'])):
        dist.append(contents['Earthquake_MAF'][scen]['SiteSourceDistance'])
    newdist = np.zeros(shape=(np.shape(dist)[0], np.shape(dist)[1]))
    # newdist = np.zeros(shape=np.shape(dist))
    for scen in range(len(dist)):
        for loc in range(np.shape(dist)[1]):
            newdist[scen, loc] = dist[scen][loc]

    dists = []
    for i in dist:
        dists.append(i[0])
    return dists, newdist

def get_pM(imdir):
    # function to get a list of the magnitudes of the events from the output of R2D
    # input: imdir is the directory to the output R2D folder
    # output: mags is a list of the magnitudes of each scenario
    with open(imdir + 'SiteIM.json', 'r') as j:
        contents = json.loads(j.read())

    pMs = []
    for scen in range(len(contents['Earthquake_MAF'])):
        pMs.append(contents['Earthquake_MAF'][scen]['MeanAnnualRate'])

    return pMs

def get_pM_pypsha(event_path):
    # function to get a np array of annual rates of events from output of pypsha
    # input the path to the event_set.pickle file
    # output numpy array of probability values (pM)
    with open(event_path,'rb') as handle:
        event_set = pickle.load(handle)
    pM = np.array(event_set.events.metadata['annualized_rate'])
    return pM

def collapse_mags(dflpi):
    out = pd.DataFrame()
    for rise in np.unique(dflpi['slr']):
        for bh in np.unique(dflpi['bh']):
            temp = dflpi[['avg', 'mags']][(dflpi['bh'] == bh) & (dflpi['slr'] == rise)]
            for M in np.unique(dflpi['mags']):
                new = temp[temp['mags'] == M]

                dicttmp = {'avg': new.mean()['avg'], 'mags': new.mean()['mags'], 'bh': bh, 'slr': rise}
                trial = pd.DataFrame(dicttmp, index=[0])
                out = pd.concat([out, trial], ignore_index=True)
    return out

def make_gdf(dflpi, cpt):
    #     names = list(cpt.keys())
    lats = np.zeros(len(dflpi))
    lons = np.zeros(len(dflpi))
    for row in range(len(dflpi)):
        name = dflpi['bh'][row]
        lats[row] = cpt[name]['Lat']
        lons[row] = cpt[name]['Lon']

    out = gpd.GeoDataFrame(dflpi, geometry=gpd.points_from_xy(lons, lats))

    return out

def restruct2(coll_dflpi):
    new = pd.DataFrame(columns=['mags', 'bh', 'geometry'])
    flag = 1
    for rise in np.unique(coll_dflpi['slr']):
        temp = coll_dflpi[coll_dflpi['slr'] == rise]
        temp.rename(columns={'avg': rise}, inplace=True)
        temp.drop(['slr'], axis=1, inplace=True)

        if flag == 1:
            #         new = pd.concat([new,temp],join = 'outer')
            new = temp
            flag = 0
        else:
            temp.reset_index(drop=True)
            new[rise] = list(temp[rise])
    return new

def rem_nan(dflpi, cpt):
    # Remove rows of dflpi with boreholes which have nan for the depth to water level
    names = list(cpt.keys())
    for bh in names:
        if np.isnan(cpt[bh]['Water depth']):
            dflpi = dflpi[dflpi['bh'] != bh]
    return dflpi

def probs(lpis):
    # function to return probability of liquefaction given
    prob = {}
    for slr in list(lpis.keys()):
        prob[slr] = {}
        for bh in list(lpis[slr].keys()):
            prob[slr][bh] = list(map(lambda x: 1 / (1 + (np.exp(-(0.218 * x - 3.092)))), lpis[slr][bh]))
    return prob

def restruct3(prob, store):
    # take probs and make single dataframe with rows as locs and P_L for each simulation
    init = list(store.keys())[0]
    vals = np.zeros(shape=(len(store[init]), len(store)))
    names = []
    for ind, sim in enumerate(store):
        lat = []
        lon = []
        names.append('sim'+str(sim))
        for loc in prob[ind][sim]:
            lat.append(store[sim][int(loc)]['Lat'])
            lon.append(store[sim][int(loc)]['Lon'])
            vals[int(loc), ind] = prob[ind][sim][loc]
    dfp = pd.DataFrame(vals, columns=names)
    dfp['lat'] = lat
    dfp['lon'] = lon
    gdfp = gpd.GeoDataFrame(dfp, geometry=gpd.points_from_xy(dfp['lon'], dfp['lat']))

    return gdfp

def get_pliq(output):
    pliq = np.zeros(shape=output.shape)
    for sim in range(len(output)):
        temp = list(map(lambda x: 1 / (1 + (np.exp(-(0.218 * x - 3.092)))), output[sim]))

        pliq[sim] = temp

    return np.array(pliq)

def ca_cl(year):
    # Function ca_cl to determine the code level of California buildings based on year of construction
    # Input: year- year of construction
    # Output: cl- code level, out of 'HC' high code, 'MC' medium code, or 'PC' pre-code

    if year <= 1940:
        cl = 'PC'
    elif year <= 1973:
        cl = 'MC'
    elif year > 1973:
        cl = 'HC'
    else:
        print('Invalid construction year')
        cl = np.nan

    return cl

def type_year(OCC, year):
    # type_year function to generate a structural type based on year and occupancy class, assuming California,
    # low rise, from Table A-2 to A-4, A-17 to A-19 in Hazus Technical Inventory

    n = random.random()
    if OCC == 'RES1':
        if n < 0.99:
            STR = 'W1'
        else:
            STR = 'RM1L'
    elif OCC == 'RES3':
        if year <= 1950:
            if n < 0.73:
                STR = 'W1'
            elif n < 0.74:
                STR = 'S1L'
            elif n < 0.75:
                STR = 'S2L'
            elif n < 0.76:
                STR = 'S3'
            elif n < 0.82:
                STR = 'S5L'
            elif n < 0.85:
                STR = 'C2L'
            elif n < 0.88:
                STR = 'C3L'
            elif n < 0.89:
                STR = 'RM1L'
            elif n < 0.98:
                STR = 'URML'
            else:
                STR = 'MH'
        elif year <= 1970:
            if n < 0.72:
                STR = 'W1'
            elif n < 0.73:
                STR = 'S1L'
            elif n < 0.75:
                STR = 'S2L'
            elif n < 0.77:
                STR = 'S3'
            elif n < 0.78:
                STR = 'S5L'
            elif n < 0.84:
                STR = 'C2L'
            elif n < 0.86:
                STR = 'C3L'
            elif n < 0.94:
                STR = 'RM1L'
            elif n < 0.97:
                STR = 'URML'
            else:
                STR = 'MH'
        elif year > 1970:
            if n < 0.73:
                STR = 'W1'
            elif n < 0.75:
                STR = 'S3'
            elif n < 0.78:
                STR = 'S4L'
            elif n < 0.84:
                STR = 'C2L'
            elif n < 0.85:
                STR = 'C3L'
            elif n < 0.86:
                STR = 'PC2L'
            elif n < 0.95:
                STR = 'RM1L'
            else:
                STR = 'MH'

    return STR

def get_theta(STR, cl):
    # getting the median of the lognormally distributed fragility function from design code level and
    # structural type, based on Hazus Earthquake technical manual Tables 5-28 to 5-32
    # output the median (theta) value
    if cl == 'PC':
        if STR == 'W1':
            theta = [0.18,0.29,0.51,0.77]
        elif STR == 'S1L':
            theta = [0.09, 0.13, 0.22, 0.38]
        elif STR == 'S2L':
            theta = [0.11, 0.14, 0.23, 0.39]
        elif STR == 'S3':
            theta = [0.08, 0.10, 0.16, 0.30]
        elif STR == 'S5L':
            theta = [0.11, 0.14, 0.22, 0.37]
        elif STR == 'C2L':
            theta = [0.11, 0.15, 0.24, 0.42]
        elif STR == 'C3L':
            theta = [0.10, 0.14, 0.21, 0.35]
        elif STR == 'RM1L':
            theta = [0.13, 0.16, 0.24, 0.43]
        elif STR == 'URML':
            theta = [0.13, 0.17, 0.26, 0.37]
        elif STR == 'MH':
            theta = [0.08, 0.11, 0.18, 0.34]
    elif cl == 'MC':
        if STR == 'W1':
            theta = [0.24,0.43,0.91,1.34]
        elif STR == 'S1L':
            theta = [0.15, 0.22, 0.42, 0.80]
        elif STR == 'S2L':
            theta = [0.20, 0.26, 0.46, 0.84]
        elif STR == 'S3':
            theta = [0.13, 0.19, 0.33, 0.60]
        elif STR == 'S4L':
            theta = [0.24, 0.39, 0.71, 1.33]  # from HC
        elif STR == 'S5L':
            theta = [0.13, 0.17, 0.28, 0.45]  # from LC
        elif STR == 'C2L':
            theta = [0.18, 0.30, 0.49, 0.87]
        elif STR == 'C3L':
            theta = [0.12, 0.17, 0.26, 0.44]  # from LC
        elif STR == 'PC2L':
            theta = [0.24, 0.36, 0.69, 1.23]  # from HC
        elif STR == 'RM1L':
            theta = [0.22, 0.30, 0.50, 0.85]
        elif STR == 'URML':
            theta = [0.14, 0.20, 0.26, 0.46]  # from LC
        elif STR == 'MH':
            theta = [0.11, 0.18, 0.31, 0.60]
    elif cl == 'HC':
        if STR == 'W1':
            theta = [0.26,0.55,1.28,2.01]
        elif STR == 'S3':
            theta = [0.15, 0.26, 0.54, 1.00]
        elif STR == 'S4L':
            theta = [0.24, 0.39, 0.71, 1.33]
        elif STR == 'C2L':
            theta = [0.24, 0.45, 0.90, 1.55]
        elif STR == 'C3L':
            theta = [0.12, 0.17, 0.26, 0.44]  # from LC
        elif STR == 'PC2L':
            theta = [0.24, 0.36, 0.69, 1.23]
        elif STR == 'RM1L':
            theta = [0.30, 0.46, 0.93, 1.57]
        elif STR == 'MH':
            theta = [0.11, 0.18, 0.31, 0.60]

    return np.array(theta)

def ff_gen(STR, PGA, cl):
    # function ff_gen to generate the fragility function of each structural type and return the
    # DS damage state of the building
    beta = 0.64
    theta = get_theta(STR, cl)
    cdf = stats.lognorm(s=beta, scale=theta).cdf(PGA)
    pdf = np.append([1], cdf) - np.append(cdf, [0])
    DS = np.random.choice([0, 1, 2, 3, 4], p=pdf)

    return DS

def loss_from_damage_single(DS):
    # function to generate loss ratio from earthquake (Hazus) damage state
    # input DS corresponds to 1:slight, 2:moderate, 3:severe, 4:complete
    if DS == 1:
        LR = 0.02
    elif DS == 2:
        LR = 0.1
    elif DS == 3:
        LR = 0.5
    elif DS == 4:
        LR = 1
    else:
        LR = 0 # DS == 0

    return LR

def loss_from_damage(DS):
    ## Loss ratio from damage state given that DS is a dataframe
    mapping = {0:0,1:0.02,2:0.1,3:0.5,4:1.0}
    LR = DS.map(mapping)
    
    return LR

def eq_shaking_loss(bldgs, bldPGA):
    LRs = np.zeros(shape=bldPGA.shape)
    cl = []
    STR = []
    for i in range(len(bldgs)):
        cl.append(ca_cl(bldgs['YearBuilt'][i]))
        STR.append(type_year(bldgs['OccupancyClass'][i], bldgs['YearBuilt'][i]))
    # STR = np.array(STR)
    # cl = np.array(cl)
    for (i, j) in itertools.product(range(len(bldgs)), range(np.shape(bldPGA)[0])):
        DS = ff_gen(STR[i], bldPGA[j, i], cl[i])
        LRs[j, i] = (loss_from_damage(DS))

    return LRs
