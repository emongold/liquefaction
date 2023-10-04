# script to plot outputs of the liquefaction postprocessing
# Emily Mongold, 2022

from .base import *
from .moss_calcs import *
from .boulangeridriss_calcs import *

def liqcalc(sim, fun, store, slr, mags, pgas, C_FC, SLR):
    lpi = {}
    if fun[sim] == 0:  # B&I
        soil, table = soil_stress(store[sim], slr)
        lpi[sim] = bi_lpi(soil, mags[sim], pgas, C_FC[sim], sim)

    elif fun[sim] == 1:  # Moss
        soil, table = soil_stress(store[sim], slr)
        depth, FS = solve_FS(soil, mags[sim], pgas, slr, sim)
        lpi[sim] = solve_LPI(depth, FS, table)[SLR[0]]

    else:
        return 'Problem with fun flag'

    return lpi

def get_sgems_sims(soildir, nsim, shape = [100, 60, 20],start = 0):
    # Function to get the data from sgems output files
    # input soildir = directory containing sgems outputs
    # input nsim = number of MC simulations
    # optional inputs shape = dimensions of soil grid and start = initial simulation
    # output fs_data = sleeve friction simulations
    # output qc_data = tip resistance simulations

    sgem_fs_sims = {}
    sgem_qc_sims = {}
    for i in range(start, nsim):
        with open(soildir + 'fs_' + str(i)) as f:
            temp = f.readlines()
        sgem_fs_sims[i] = [float(line.rstrip(" \n")) for line in temp[3:]]
        with open(soildir + 'qc_' + str(i)) as f:
            temp = f.readlines()
        sgem_qc_sims[i] = [float(line.rstrip(" \n")) for line in temp[3:]]

    fs_data = {}
    qc_data = {}
    for i in range(start, nsim):
        fs_data[i] = np.reshape(sgem_fs_sims[i], shape, order='F')
        qc_data[i] = np.reshape(sgem_qc_sims[i], shape, order='F')

    return fs_data, qc_data

def get_points(fs_data, qc_data, nsim, geoplot, utmX0=558700, utmY0=4178000, width=100, shape = [100, 60, 20], start = 0):
    # function to get the soil data at each point
    # inputs fs_data and qc_data from sgems data, restructured
    # input nsim = number of MC simulations
    # input geoplot = boundary of points to cutoff from grid
    # optional inputs shape = dimensions of soil grid, start = starting simulation
    # ouput points = geodataframe of points in input geojson boundary

    xs = range(shape[0])
    x = nm.repmat(xs, 1, shape[1])
    indX = list(x[0])

    indY = []
    y = []
    for i in range(shape[1]):
        y.append(np.full(shape[0], i))
    for i in range(len(y)):
        for j in list(y[i]):
            indY.append(j)

    # Changing these since width is changing
    # y = list(map(lambda x: x / 10, indY))
    # x = list(map(lambda x: x / 10, indX))

    # utmX = list(map(lambda x: (width * x) + utmX0, x))
    # utmY = list(map(lambda x: (width * x) + utmY0, y))

    utmX = list(map(lambda x: (width * x) + utmX0, indX))
    utmY = list(map(lambda x: (width * x) + utmY0, indY))

    grid = pd.DataFrame(columns=['y', 'x'])
    # grid['y'] = y
    # grid['x'] = x
    grid['y'] = indY
    grid['x'] = indX
    LAT = np.zeros(len(utmX))
    LON = np.zeros(len(utmY))
    for i in range(len(utmX)):
        lat, lon = utm.to_latlon(utmX[i], utmY[i], 10, northern=True)  # only works for 10N region (CA/Bay area?)
        LAT[i] = lat
        LON[i] = lon

    grid['lat'] = LAT
    grid['lon'] = LON
    grid['utmX'] = utmX
    grid['utmY'] = utmY
    grid['indX'] = indX
    grid['indY'] = indY

    for real in range(start, nsim):
        temp = []
        for i in range(len(grid)):
            temp.append(fs_data[real][grid['indX'][i], grid['indY'][i], :])

        grid['fs_' + str(real)] = temp

        temp = []
        for i in range(len(grid)):
            temp.append(qc_data[real][grid['indX'][i], grid['indY'][i], :])

        grid['qc_' + str(real)] = temp

    gdf = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid['lon'], grid['lat']))

    gdf.set_crs(epsg=4326, inplace=True)
    Alameda = gpd.read_file(geoplot)
    points = gpd.sjoin(gdf, Alameda)
    points.reset_index(inplace=True, drop=True)

    return points

def setup_soil(points, nsim, start = 0):
    # Function to format and store the simulated soil data
    # input points = data from grid generation of soil
    # input nsim = number of MC simulations
    # optional input start = starting simulation
    # output store = dataframe containing the soil for each simulation

    g = 9.81
    Pa = 0.101325  # MPa
    rho_w = 1  # Mg/m^3
    gamma_w = rho_w * g / 1000  # MPa/m
    store = {}
    for sim in range(start, nsim):
        soil = {}
        for i, point in enumerate(points):
            soil[i] = {
                'Lat': point['lat'],
                'Lon': point['lon'],
                'utmX': point['utmX'],
                'utmY': point['utmY'],
                'elev': point['elev'],
                'CPT_data': pd.DataFrame({
                    'start': range(19, -1, -1),
                    'q_c': point['qc_' + str(sim)],
                    'f_s': point['fs_' + str(sim)],
                    'd': np.arange(19, -1, -1) + 0.5,
                    'dz': 1
                })
            }
            soil[i]['CPT_data']['R_f'] = 100 * soil[i]['CPT_data']['f_s'] / soil[i]['CPT_data']['q_c']

            gamma = np.zeros(len(soil[i]['CPT_data']))

            for j in range(len(soil[i]['CPT_data'])):
                # Calculating soil unit weight from Robertson and Cabal (2010)
                if soil[i]['CPT_data']['R_f'][j] == 0:
                    if soil[i]['CPT_data']['q_c'][j] == 0:
                        gamma[j] = gamma_w * 1.236
                    else:
                        gamma[j] = gamma_w * (0.36 * np.log10(soil[i]['CPT_data']['q_c'][j] / Pa) + 1.236)
                elif soil[i]['CPT_data']['q_c'][j] == 0:
                    gamma[j] = gamma_w * (0.27 * (np.log10(soil[i]['CPT_data']['R_f'][j])) + 1.236)
                else:
                    gamma[j] = gamma_w * (0.27 * (np.log10(soil[i]['CPT_data']['R_f'][j])) +
                                          0.36 * np.log10(soil[i]['CPT_data']['q_c'][j] / Pa) + 1.236)

            soil[i]['CPT_data']['gamma'] = gamma
            soil[i]['CPT_data']['dsig_v'] = soil[i]['CPT_data']['dz'] * soil[i]['CPT_data']['gamma']
        store[sim] = soil

    return store

def interp_elev(points, geotiff_patha, geotiff_pathb):
    # Function to interpolate the elevation to each point
    # input points = dataframe with points of interest
    # input geotiff_path{a&b} = directories containing geotiff elevation files for area
    # output points = updated dataframe with elevation at each point

    # Open into an xarray.DataArray
    geotiff_da = xr.open_rasterio(geotiff_patha)
    geotiff_db = xr.open_rasterio(geotiff_pathb)

    # Covert our xarray.DataArray into a xarray.Dataset
    geotiff_dsa = geotiff_da.to_dataset('band')
    geotiff_dsb = geotiff_db.to_dataset('band')

    # Rename the variable to a more useful name
    geotiff_dsa = geotiff_dsa.rename({1: 'elev'})
    geotiff_dsb = geotiff_dsb.rename({1: 'elev'})

    ymax = points['utmY'].max() + 100
    xmax = points['utmX'].max() + 100
    xmin = points['utmX'].min() - 100

    b = geotiff_dsb.elev.where(geotiff_dsb.elev > 0)
    b = b.where(b.y < ymax)
    b = b.where(b.x > xmin)
    a = geotiff_dsa.elev.where(geotiff_dsa.elev > 0)  # run this with path to g2
    a = a.where(a.y < ymax)
    a = a.where(a.x < xmax)

    a = a.dropna(dim="y", how="all")
    a = a.dropna(dim="x", how="all")
    b = b.dropna(dim="y", how="all")
    b = b.dropna(dim="x", how="all")

    # Za = a.to_numpy()
    # Zb = b.to_numpy()
    Za = a.values
    Zb = b.values

    Xa = np.array(a.x)
    Xb = np.array(b.x)
    Ya = np.array(a.y)
    Yb = np.array(b.y)

    X = []
    Y = []
    Z = []

    for (x_a, y_a) in itertools.product(range(len(Xa)), range(len(Ya))):
        if pd.notna(Za[y_a, x_a]) and Za[y_a, x_a] > 1e-30:
            X.append(Xa[x_a])
            Y.append(Ya[y_a])
            Z.append(Za[y_a, x_a])
    for (x_b, y_b) in itertools.product(range(len(Xb)), range(len(Yb))):
        if pd.notna(Zb[y_b, x_b]) and Zb[y_b, x_b] > 1e-30:
            X.append(Xb[x_b])
            Y.append(Yb[y_b])
            Z.append(Zb[y_b, x_b])

    elevs = NNR(np.array([np.array(points['utmX']), np.array(points['utmY'])]).T, np.array([X, Y]).T,
                np.array(Z), sample_size=-1, n_neighbors=4, weight='distance2')
    points['elev'] = elevs

    return points

def save_run(outdir,pgas,C_FC,wd_var,fun,mags,gdfp):
    # Function to store the outputs of MC sims
    # inputs are the outdir = directory of where to store outputs
    # input pgas, C_FC, wd_var, fun, mags = the varied input parameters
    # input gdfp = the output geodataframe of lpi values
    new = {}
    for i in range(len(pgas)):
        new[i] = np.zeros(len(pgas[i]))
        for j in range(len(pgas[i])):
            new[i][j] = pgas[i][j]#[0]
    nsim = len(pgas[0])
    temp = pd.DataFrame(new)
    temp.to_csv(outdir + 'pga.csv')

    np.savetxt(outdir + 'cfc.csv', C_FC, delimiter=",")
    np.savetxt(outdir + 'wd.csv', wd_var, delimiter=",")
    np.savetxt(outdir + 'fun.csv',fun,delimiter=",")
    np.savetxt(outdir + 'M.csv',mags[0:nsim],delimiter=",")

    gdfp.to_file(outdir +'/gdfp.geojson', driver="GeoJSON")

    return

def grid_soil(datadir, outdir, utmX0=558700, utmY0=4178000, width=100, outfile='gridded_data.csv',
              datafile='point_data.dat'):
    # function grid_soil to convert data to x,y values for input to SGeMS
    # input datadir = directory to the USGS cpt data
    # input outdir = directory to output the .csv file
    # input utmX0 = minimum utmX value to set as new zero (default Alameda)
    # input utmY0 = minimum utmY value to set as new zero (default Alameda)

    d = {}
    names = []
    info = {}
    gwt = []

    for filename in filter(lambda x: x[-4:] == '.txt', os.listdir(datadir)):

        with open(os.path.join(os.getcwd(), datadir, filename)) as f:
            name = datadir + filename
            df_temp = pd.read_csv(name, delimiter="\s+", skiprows=17)
            df_temp = df_temp.dropna(axis='columns', how='all')
            df_temp.columns = ['Depth', 'Tip_Resistance', 'Sleeve_Friction', 'Inclination', 'Swave_travel_time']
            df_temp = df_temp[-((df_temp['Sleeve_Friction'] < 0) | (df_temp['Tip_Resistance'] < 0))]
            df_temp = df_temp[df_temp['Depth'] <= 20]

            key = list(dict(l.strip().rsplit(maxsplit=1) for l in open(name) \
                            if any(l.strip().startswith(i) for i in ('File name:'))).values())[0]
            names.append(key)
            d[key] = dict(l.strip().rsplit('\t', maxsplit=1) for l in open(name) \
                          if
                          (any(l.strip().startswith(i) for i in ('"UTM-X', '"UTM-Y', '"Elev', '"Water depth', 'Date')) \
                           and len(l.strip().rsplit('\t', maxsplit=1)) == 2))
            info[key] = {}
            info[key]['CPT_data'] = df_temp
            info[key]['CPT_data']['Sleeve_Friction'] = info[key]['CPT_data'][
                                                           'Sleeve_Friction'] / 1000  # convert to units of MPa

            for i in d[key]:
                if i.startswith('"UTM-X'):
                    info[key]['UTM-X'] = int(d[key][i])
                elif i.startswith('"UTM-Y'):
                    info[key]['UTM-Y'] = int(d[key][i])
                elif i.startswith('"Elev'):
                    info[key]['Elev'] = float(d[key][i])
                elif i.startswith('"Water depth'):
                    info[key]['Water depth'] = float(d[key][i])
                    gwt.append(float(d[key][i]))
                elif i.startswith('Date'):
                    info[key]['Date'] = datetime.datetime.strptime(d[key][i], '%m/%d/%Y')

            if not 'Elev' in info[key]:
                info[key]['Elev'] = np.nan
            if not 'Water depth' in info[key]:
                info[key]['Water depth'] = np.nan

            for key in names:
                info[key]['x'] = (info[key]['UTM-X'] - utmX0) / width
                info[key]['y'] = (info[key]['UTM-Y'] - utmY0) / width

            for key in names:
                z = range(1, int(max(info[key]['CPT_data']['Depth'])) + 1)
                fs = []
                qc = []
                de = []
                for dep in z:
                    fstemp = info[key]['CPT_data']['Sleeve_Friction'][
                        (info[key]['CPT_data']['Depth'] <= dep) & (info[key]['CPT_data']['Depth'] > dep - 1)]
                    qctemp = info[key]['CPT_data']['Tip_Resistance'][
                        (info[key]['CPT_data']['Depth'] <= dep) & (info[key]['CPT_data']['Depth'] > dep - 1)]

                    fs.append(np.mean(fstemp))
                    qc.append(np.mean(qctemp))
                    de.append(20 - dep)
                info[key]['z'] = de
                info[key]['fs'] = fs
                info[key]['qc'] = qc

            output = pd.DataFrame(columns=['x', 'y', 'z', 'fs', 'qc'])
            for key in names:
                tempdf = pd.DataFrame()
                tempdf['z'] = info[key]['z']
                tempdf['x'] = info[key]['x']
                tempdf['y'] = info[key]['y']
                tempdf['fs'] = info[key]['fs']
                tempdf['qc'] = info[key]['qc']
                output = pd.concat([output, tempdf], ignore_index=True)

    output.to_csv(outdir + outfile, index=False)

    with open(outdir + datafile, "w") as txtfile:
        txtfile.write('points \n')
        txtfile.write(str(len(output.keys())) + '\n')
        for i in output.keys():
            txtfile.write(i + '\n')
        for i in range(len(output)):
            if i == 0:
                cnt = 1
            else:
                cnt = 0
            for j in list(map(lambda x: str(x), list(output.loc[i]))):
                if cnt == 0:
                    txtfile.write('\n' + j + ' ')
                    cnt += 1
                else:
                    txtfile.write(j + ' ')

    return

def save_ins(outdir, pgas, C_FC, wd_var, fun, mags, store):
    # Function to store the inputs of MC sims before lpi calcs (to split simulations in later step)
    # inputs are the outdir = directory of where to store outputs
    # input pgas, C_FC, wd_var, fun, mags = the varied input parameters
    # input store = the temporary geodataframe of soil values
    new = {}
    for i in range(len(pgas)):
        new[i] = np.zeros(len(pgas[i]))
        for j in range(len(pgas[i])):
            new[i][j] = pgas[i][j]  # [0]
    nsim = len(pgas[0])
    temp = pd.DataFrame(new)
    temp.to_csv(outdir + 'pga.csv')

    np.savetxt(outdir + 'cfc.csv', C_FC, delimiter=",")
    np.savetxt(outdir + 'wd.csv', wd_var, delimiter=",")
    np.savetxt(outdir + 'fun.csv', fun, delimiter=",")
    np.savetxt(outdir + 'M.csv', mags[0:nsim], delimiter=",")

    store.to_file(outdir + '/store.geojson', driver="GeoJSON")

    return

def save_part(outdir, gdfp, sim):
    # Function to store the outputs of MC sims when separated in parts
    # input gdfp = the output geodataframe of lpi values

    gdfp.to_file(outdir + '/gdfp' + str(sim) + '.geojson', driver="GeoJSON")

    return
