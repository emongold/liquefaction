# script to hold functions for liquefaction preprocessing
# Emily Mongold, 2022

from .base import *


def get_wd(datadir, points):
    d = {}
    names = []
    info = {}

    for filename in filter(lambda x: x[-4:] == '.txt', os.listdir(datadir)):

        with open(os.path.join(os.getcwd(), datadir, filename)) as f:
            name = datadir + filename
            key = list(dict(l.strip().rsplit(maxsplit=1) for l in open(name) \
                            if any(l.strip().startswith(i) for i in ('File name:'))).values())[0]
            names.append(key)
            d[key] = dict(l.strip().rsplit('\t', maxsplit=1) for l in open(name) \
                          if
                          (any(l.strip().startswith(i) for i in ('"UTM-X', '"UTM-Y', '"Elev', '"Water depth', 'Date')) \
                           and len(l.strip().rsplit('\t', maxsplit=1)) == 2))
            info[key] = {}
            for i in d[key]:
                if i.startswith('"UTM-X'):
                    info[key]['UTM-X'] = int(d[key][i])
                elif i.startswith('"UTM-Y'):
                    info[key]['UTM-Y'] = int(d[key][i])
                elif i.startswith('"Water depth'):
                    info[key]['Water depth'] = float(d[key][i])
                elif i.startswith('"Elev'):
                    info[key]['Elev'] = float(d[key][i])

            info[key]['x'] = (info[key]['UTM-X'] - 559000) / 1000
            info[key]['y'] = (info[key]['UTM-Y'] - 4178000) / 1000

            if not 'Water depth' in info[key]:
                del (info[key])

    del (d)

    X = []
    Y = []
    GWT = []
    for i in info:
        X.append(info[i]['UTM-X'])
        Y.append(info[i]['UTM-Y'])
        GWT.append(info[i]['Elev'] - info[i]['Water depth'])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(GWT)

    gwt = NNR(np.array([points['utmX'], points['utmY']]).T, np.array([X, Y]).T, Z, sample_size=-1, n_neighbors=4,
              weight='distance2')

    wd = points['elev'] - gwt

    return wd

def vary_wd(wd_base, wd_var, store, start = 0, nsim = 0):
    # for sim in range(len(wd_var)):
    if nsim == 0:
        nsim = len(wd_var)
    for sim in range(start, nsim):

        wd = list(map(lambda x: x + wd_var[sim], wd_base))

        for loc in range(len(wd)):
            store[sim][loc]['Water depth'] = wd[loc]

    return store

def load_cpt_data(datadir):
    # input: datadir is the directory path to the cpt data
    # output: cpt is a dictionary with an np directory for each borehole set of cpt data

    d = {}
    names = []
    cpt = {}
    # Constants
    g = 9.81
    Pa = 0.101325  # MPa
    rho_w = 1  # Mg/m^3
    gamma_w = rho_w * g / 1000  # MPa

    for filename in filter(lambda x: x[-4:] == '.txt', os.listdir(datadir)):

        with open(os.path.join(datadir, filename)) as f:
            name = datadir + filename
            df_temp = pd.read_csv(name, delimiter="\s+", skiprows=17)
            df_temp = df_temp.dropna(axis='columns', how='all')
            df_temp.columns = ['Depth', 'Tip_Resistance', 'Sleeve_Friction', 'Inclination', 'Swave_travel_time']
            df_temp = df_temp[-((df_temp['Sleeve_Friction'] < 0) | (df_temp['Tip_Resistance'] < 0))]

            df_temp = df_temp[df_temp['Depth'] <= 20]
            df_temp['Sleeve_Friction'] = df_temp['Sleeve_Friction'] / 1000  # convert to units of MPa

            temp = pd.DataFrame(np.zeros(shape=(len(df_temp), 7)),
                                columns=['start', 'q_c', 'f_s', 'd', 'dz', 'gamma', 'R_f'])
            temp['q_c'][0] = df_temp['Tip_Resistance'][0] / 2
            temp['f_s'][0] = df_temp['Sleeve_Friction'][0] / 2
            temp['d'][0] = np.average([temp['start'][0], df_temp['Depth'][0]])
            temp['dz'][0] = df_temp['Depth'][0] - temp['start'][0]
            temp['R_f'][0] = 100 * temp['f_s'][0] / temp['q_c'][0]
            temp['gamma'][0] = gamma_w * (0.27 * (np.log10(temp['R_f'][0])) +
                                          0.36 * np.log10(temp['q_c'][0] / Pa) + 1.236)
            for i in range(1, len(df_temp)):
                temp['start'][i] = df_temp['Depth'].iloc[i - 1]
                temp['f_s'][i] = np.average([df_temp['Sleeve_Friction'].iloc[i], df_temp['Sleeve_Friction'].iloc[i - 1]])
                temp['q_c'][i] = np.average([df_temp['Tip_Resistance'].iloc[i], df_temp['Tip_Resistance'].iloc[i - 1]])
                temp['d'][i] = np.average([temp['start'][i], df_temp['Depth'].iloc[i]])
                temp['dz'][i] = df_temp['Depth'].iloc[i] - temp['start'][i]
                temp['R_f'][i] = 100 * temp['f_s'][i] / temp['q_c'][i]
                # Calculating soil unit weight from Robertson and Cabal (2010)
                if temp['R_f'][i] == 0:
                    if temp['q_c'][i] == 0:
                        temp['gamma'][i] = gamma_w * 1.236
                    else:
                        temp['gamma'][i] = gamma_w * (0.36 * np.log10(temp['q_c'][i] / Pa) + 1.236)
                elif temp['q_c'][i] == 0:
                    temp['gamma'][i] = gamma_w * (0.27 * (np.log10(temp['R_f'][i])) + 1.236)
                else:
                    temp['gamma'][i] = gamma_w * (0.27 * np.log10(temp['R_f'][i]) +
                                                  0.36 * np.log10(temp['q_c'][i] / Pa) + 1.236)

            temp['dsig_v'] = temp['dz'] * temp['gamma']

            key = list(dict(l.strip().rsplit(maxsplit=1) for l in open(name) if any(l.strip().startswith(i) for i in 'File name:')).values())[0]
            names.append(key)
            d[key] = dict(l.strip().rsplit('\t', maxsplit=1) for l in open(name) \
                          if (any(l.strip().startswith(i) for i in ('"UTM-X', '"UTM-Y', '"Elev', '"Water depth', 'Date')) and len(l.strip().rsplit('\t', maxsplit=1)) == 2))

            cpt[key] = {}
            cpt[key]['CPT_data'] = temp

            for i in d[key]:
                if i.startswith('"UTM-X'):
                    cpt[key]['UTM-X'] = int(d[key][i])
                elif i.startswith('"UTM-Y'):
                    cpt[key]['UTM-Y'] = int(d[key][i])
                elif i.startswith('"Elev'):
                    cpt[key]['Elev'] = float(d[key][i])
                elif i.startswith('"Water depth'):
                    cpt[key]['Water depth'] = float(d[key][i])
                elif i.startswith('Date'):
                    cpt[key]['Date'] = datetime.datetime.strptime(d[key][i], '%m/%d/%Y')

            if 'Elev' not in cpt[key]:
                cpt[key]['Elev'] = np.nan
            if 'Water depth' not in cpt[key]:
                cpt[key]['Water depth'] = np.nan

    for i in range(len(names)):
        cpt[names[i]]['Lat'], cpt[names[i]]['Lon'] = utm.to_latlon(cpt[names[i]]['UTM-X'], cpt[names[i]]['UTM-Y'], 10,
                                                                   northern=True)
    return cpt

def get_mags(imdir):
    # function to get a list of the magnitudes of the events from the output of R2D
    # input: imdir is the directory to the output R2D folder
    # output: mags is a list of the magnitudes of each scenario
    with open(imdir + 'SiteIM.json', 'r') as j:
        contents = json.loads(j.read())

    mags = []
    for scen in range(len(contents['Earthquake_MAF'])):
        mags.append(contents['Earthquake_MAF'][scen]['Magnitude'])

    return mags

def get_pgas(imdir, num, names, gmm):
    # get an array of all the pgas of the
    # input: imdir is the directory to the output R2D folder, in a list
    # input: num is the number of scenarios (length of mags vector)
    # input: names is the keys of the boreholes (list((cpt.keys()))
    # output: pgas is a dict with keys of bh names and an n_scen x n_sim array for each

    with open(os.path.join(imdir[0], 'EventGrid.csv')) as f:
        data = pd.read_csv(f)
    pgas = {}
    for i in range(len(names)):
        file = str(data['GP_file'][names[i]])
        pgas[i] = []
        for scen in range(num):
            direc = imdir[int(gmm[scen])]
            gms = pd.read_csv(direc + 'scenario' + str(scen + 1) + '/' + file)
            pgas[i].append(gms['PGA'])
        pgas[i] = np.array(pgas[i])
    return pgas

def interp_gwt(cpt):
    # fill in missing depth to groundwater values
    # input: cpt dict containing dataframes of cpt data fro each borehole
    # output: cpt dict with the same information but replaced NaN values under 'Water depth'
    names = list(cpt.keys())
    boreholes = pd.DataFrame(columns=['name'], data=names)
    la = list()
    lo = list()
    gwtemp = np.zeros(shape=(len(cpt), 1))
    for i in range(len(cpt)):
        boreholes['name'][i] = names[i]
        la.append(cpt[names[i]]['Lat'])
        lo.append(cpt[names[i]]['Lon'])
        gwtemp[i] = cpt[names[i]]['Water depth']
    boreholes['gwt'] = gwtemp
    del gwtemp
    bhgdf = gpd.GeoDataFrame(boreholes, geometry=gpd.points_from_xy(lo, la))
    fill_bhgdf = bhgdf[bhgdf['gwt'] != bhgdf['gwt']]  # Separate boreholes without groundwater level

    bhgdf.dropna(subset=["gwt"], inplace=True, )  # Remove borholes without groundwater level
    bhgdf.reset_index(inplace=True)

    # code from
    # https://hatarilabs.com/ih-en/geospatial-triangular-interpolation-with-python-scipy-geopandas-and-rasterio-tutorial
    totalPointsArray = np.zeros([bhgdf.shape[0], 3])
    for index, point in bhgdf.iterrows():
        pointArray = np.array([point.geometry.coords.xy[0][0], point.geometry.coords.xy[1][0], point['gwt']])
        totalPointsArray[index] = pointArray
    triFn = Triangulation(totalPointsArray[:, 0], totalPointsArray[:, 1])
    linTriFn = LinearTriInterpolator(triFn, totalPointsArray[:, 2])

    rasterRes = 0.0001  # using lat/lon degree values not m

    xCoords = np.arange(totalPointsArray[:, 0].min(), totalPointsArray[:, 0].max() + rasterRes, rasterRes)
    yCoords = np.arange(totalPointsArray[:, 1].min(), totalPointsArray[:, 1].max() + rasterRes, rasterRes)
    zCoords = np.zeros([yCoords.shape[0], xCoords.shape[0]])

    for indexX, x in np.ndenumerate(xCoords):
        for indexY, y in np.ndenumerate(yCoords):
            tempZ = linTriFn(x, y)
            if tempZ == tempZ:
                zCoords[indexY, indexX] = tempZ
            else:
                zCoords[indexY, indexX] = np.nan
    for index, point in fill_bhgdf.iterrows():
        tempZ = linTriFn(point.geometry.coords.xy[0][0], point.geometry.coords.xy[1][0])
        if tempZ == tempZ:
            fill_bhgdf.loc[index, 'gwt'] = float(tempZ)
        else:
            fill_bhgdf.loc[index, 'gwt'] = np.nan
    fill_bhgdf.reset_index(inplace=True)
    final_bhgdf = pd.concat([bhgdf, fill_bhgdf]).set_index('index').sort_index()

    for i in range(len(final_bhgdf)):
        cpt[final_bhgdf['name'][i]]['Water depth'] = final_bhgdf['gwt'][i]

    return cpt

def get_pgas_from_grid_r2d(imdir, nsim, names, geoplot, gmm, width = 100, utmX0=558700, utmY0=4178000, shape = [100, 60, 20]):
    pgas = {}
    with open(os.path.join(imdir[0], 'EventGrid.csv')) as f:
        data = pd.read_csv(f)
    lons = data['Longitude']
    lats = data['Latitude']
    pgaZ = {}

    for i in range(len(data['GP_file'])):
        file = str(data['GP_file'][i])
        direc = imdir[int(gmm[i])]
        gms = pd.read_csv(direc + 'IMs/' + file)  # + 'IMs/' ((was there before, think I don't need))
        pgaZ[i] = gms['PGA']
    #     nsim = len(pgaZ[0])

    utmX = np.zeros(len(lons))
    utmY = np.zeros(len(lons))
    for i in range(len(utmX)):
        (utmX[i], utmY[i], reg, northrn) = utm.from_latlon(lats[i], lons[i])

    X = (utmX - utmX0) / width
    Y = (utmY - utmY0) / width

    xs = range(shape[0])
    x = nm.repmat(xs, 1, shape[1])
    x_t = list(x[0])  # was indX

    y_t = []  # was indY
    y = []
    for i in range(shape[1]):
        y.append(np.full(shape[0], i))
    for i in range(len(y)):
        for j in list(y[i]):
            y_t.append(j)  # was indY

    # y_t = list(map(lambda x: x / 10, indY))
    # x_t = list(map(lambda x: x / 10, indX))
    utmX = list(map(lambda x: (width * x) + utmX0, x_t))
    utmY = list(map(lambda x: (width * x) + utmY0, y_t))

    grid = pd.DataFrame(columns=['y', 'x'])
    grid['y'] = y_t
    grid['x'] = x_t
    LAT = np.zeros(len(utmX))
    LON = np.zeros(len(utmY))
    for i in range(len(utmX)):
        lat, lon = utm.to_latlon(utmX[i], utmY[i], reg, northrn)
        LAT[i] = lat
        LON[i] = lon
    grid['lat'] = LAT
    grid['lon'] = LON
    # grid['indX'] = indX
    # grid['indY'] = indY

    Alameda = gpd.read_file(geoplot)
    for i in range(len(names)):
        pgas[i] = []
    for sim in range(nsim):
        Z = []
        for i in range(len(pgaZ)):  # pga at each location
            Z.append(pgaZ[i][sim])
        Z = np.array(Z)
        z_t = NNR(np.array([x_t, y_t]).T, np.array([X, Y]).T, Z, sample_size=-1, n_neighbors=4, weight='distance2')
        grid['pga'] = z_t
        gdf = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid['lon'], grid['lat']))
        points = gpd.sjoin(gdf, Alameda)
        points.reset_index(inplace=True, drop=True)
        for i in range(len(names)):
            pgas[i].append(points['pga'][i])

    for i in range(len(names)):
        pgas[i] = np.array(pgas[i])

    return pgas

def get_pgas_from_grid_par(sim, imdir, names, geoplot, width = 100, utmX0=558700, utmY0=4178000, shape = [100, 60, 20]):
    pgas = {}
    with open(os.path.join(imdir, 'EventGrid.csv')) as f:
        data = pd.read_csv(f)
    lons = data['Longitude']
    lats = data['Latitude']
    pgaZ = []

    for i in range(len(data['GP_file'])):
        file = str(data['GP_file'][i])
        gms = pd.read_csv(imdir + 'IMs/' + file)  # + 'IMs/' ((was there before, think I don't need))
        pgaZ.append(gms['PGA'][sim])
    #     nsim = len(pgaZ[0])

    utmX = np.zeros(len(lons))
    utmY = np.zeros(len(lons))
    for i in range(len(utmX)):
        (utmX[i], utmY[i], reg, northrn) = utm.from_latlon(lats[i], lons[i])

    X = (utmX - utmX0) / width
    Y = (utmY - utmY0) / width

    xs = range(shape[0])
    x = nm.repmat(xs, 1, shape[1])
    x_t = list(x[0])  # was indX

    y_t = []  # was indY
    y = []
    for i in range(shape[1]):
        y.append(np.full(shape[0], i))
    for i in range(len(y)):
        for j in list(y[i]):
            y_t.append(j)  # was indY

    # y_t = list(map(lambda x: x / 10, indY))
    # x_t = list(map(lambda x: x / 10, indX))
    utmX = list(map(lambda x: (width * x) + utmX0, x_t))
    utmY = list(map(lambda x: (width * x) + utmY0, y_t))

    grid = pd.DataFrame(columns=['y', 'x'])
    grid['y'] = y_t
    grid['x'] = x_t
    LAT = np.zeros(len(utmX))
    LON = np.zeros(len(utmY))
    for i in range(len(utmX)):
        lat, lon = utm.to_latlon(utmX[i], utmY[i], reg, northrn)
        LAT[i] = lat
        LON[i] = lon
    grid['lat'] = LAT
    grid['lon'] = LON
    # grid['indX'] = indX
    # grid['indY'] = indY

    Alameda = gpd.read_file(geoplot)
    for i in range(len(names)):
        pgas[i] = []
    # Z = []
    # for i in range(len(pgaZ)):  # pga at each location
    #     Z.append(pgaZ[i][sim])
    Z = np.array(pgaZ)
    z_t = NNR(np.array([x_t, y_t]).T, np.array([X, Y]).T, Z, sample_size=-1, n_neighbors=4, weight='distance2')
    grid['pga'] = z_t
    gdf = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid['lon'], grid['lat']))
    points = gpd.sjoin(gdf, Alameda)
    points.reset_index(inplace=True, drop=True)
    for i in range(len(names)):
        pgas[i].append(points['pga'][i])

    for i in range(len(names)):
        pgas[i] = np.array(pgas[i])

    return pgas

def get_pgas_from_grid_pypsha(imdir, nsim, names, geoplot, width = 100, utmX0=558700, utmY0=4178000, shape = [100, 60, 20]):
    # imdir must be where Alameda_sites.csv and pgas_pypsha.csv are located
    pgas = {}
    site_file = imdir + "Alameda_sites.csv"
    pgaZ = np.genfromtxt(imdir + 'pgas_pypsha.csv', delimiter=',')
    data = pd.read_csv(site_file)
    lons = data['x']
    lats = data['y']

    nsim = len(pgaZ)  # changed since the shape of pgaZ is nsim x ngridloc
    utmX = np.zeros(len(lons))
    utmY = np.zeros(len(lons))
    for i in range(len(utmX)):
        (utmX[i], utmY[i], reg, northrn) = utm.from_latlon(lats[i], lons[i])

    X = (utmX - utmX0) / width
    Y = (utmY - utmY0) / width

    xs = range(shape[0])
    x = nm.repmat(xs, 1, shape[1])
    x_t = list(x[0])

    y_t = []
    y = []
    for i in range(shape[1]):
        y.append(np.full(shape[0], i))
    for i in range(len(y)):
        for j in list(y[i]):
            y_t.append(j) 
    utmX = list(map(lambda x: (width * x) + utmX0, x_t))
    utmY = list(map(lambda x: (width * x) + utmY0, y_t))

    grid = pd.DataFrame(columns=['y', 'x'])
    grid['y'] = y_t
    grid['x'] = x_t
    LAT = np.zeros(len(utmX))
    LON = np.zeros(len(utmY))
    for i in range(len(utmX)):
        lat, lon = utm.to_latlon(utmX[i], utmY[i], reg, northrn)
        LAT[i] = lat
        LON[i] = lon
    grid['lat'] = LAT
    grid['lon'] = LON

    Alameda = gpd.read_file(geoplot)
    for i in range(len(names)):
        pgas[i] = []
    for sim in range(nsim):
        Z = []
        for i in range(pgaZ.shape[1]):  # pga at each location
            Z.append(pgaZ[sim,i])
        Z = np.array(Z)
        z_t = NNR(np.array([x_t, y_t]).T, np.array([X, Y]).T, Z, sample_size=-1, n_neighbors=4, weight='distance2')
        grid['pga'] = z_t
        gdf = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid['lon'], grid['lat']))
        points = gpd.sjoin(gdf, Alameda)
        points.reset_index(inplace=True, drop=True)
        for i in range(len(names)):
            pgas[i].append(points['pga'][i])

    for i in range(len(names)):
        pgas[i] = np.array(pgas[i])

    return pgas

def get_mags_pypsha(event_path):
    # function to get a np array of magnitudes of events from output of pypsha
    # input the path to the event_set.pickle file
    # output numpy array of magnitude values (mags)
    with open(event_path,'rb') as handle:
        event_set = pickle.load(handle)
    mags = np.array(event_set.events.metadata['magnitude'])

    return mags

def run_pypsha_pgas(imdir,gmm):
    # function to obtain pgas from pypsha output file based on gmm for each scenario.
    
    with open(imdir + 'event_save.pickle','rb') as handle:
        event_set = pickle.load(handle)
    sa_intensity_ids = [item[:-4] for item in list(event_set.events.intensity_filelist['filename'])]
    # sa_intensity_ids = ["ASK2014_PGA","BSSA2014_PGA","CY2014_PGA"] # should edit to obtain automatically
    
    ask_events = event_set.maps[sa_intensity_ids[0]]
    ask_events = ask_events.reset_index(level='map_id')
    ask_events = ask_events[ask_events['map_id'] == 0]
    ask_events.drop('map_id', axis=1,inplace=True)
    
    pga = np.zeros(ask_events.shape)
    n = 0
    for scen in event_set.maps[sa_intensity_ids[0]]['site0'].keys()[:(2*len(gmm))]:  ## The 2*len(gmm) is a hack to get the right number of scenarios for now
        if scen[-1] == 0:
            mod = int(gmm[n])
            pga[n,:] = event_set.maps[sa_intensity_ids[mod]].loc[scen]
            n += 1
            
    np.savetxt(imdir + 'pgas_pypsha.csv',pga,delimiter=',')

    return

def run_pypsha(site_file,nmaps,outdir):
    # have some fixed parameters, such as attenuations and PGA as IM
    test_site = psha.PSHASite(name = 'site',
                            site_filename = site_file,
                            erf=1, intensity_measures = [1],
                            attenuations = [1,2,4],
                            overwrite=True)
    test_site.write_opensha_input(overwrite = True)
    test_site.run_opensha(overwrite= True, write_output_tofile = True)
    event_set = psha.PshaEventSet(test_site)
    # sa_intensity_ids = ["ASK2014_PGA","BSSA2014_PGA","CY2014_PGA"]
    sa_intensity_ids = [item[:-4] for item in list(event_set.events.intensity_filelist['filename'])]
    
    event_set.generate_sa_maps(sa_intensity_ids, nmaps)
    with open(outdir + 'event_save.pickle','wb') as handle:
        pickle.dump(event_set, handle)

    return

