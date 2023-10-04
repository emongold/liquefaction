## liq_setup.py
## by Emily Mongold, emongold<at>stanford<dot>edu
## Last updated 9/21/2023
## This script sets up the run of the liquefaction model.


from liquefaction import *

#### Find each of these inputs and alter the paths #####
geotiff_patha = './inputs/elev/ned03m37122g2.tif'
geotiff_pathb = './inputs/elev/ned03m37122g3.tif'
soildir = './inputs/soil/'
datadir = './inputs/CPT/'
geoplot = './inputs/alameda_plots/Alameda_shape.geojson'
imdir = './inputs/generate_IMs/'
indir = './outputs/'
outdir = './outputs/'
# soil grid properties:
# shape = [33, 20, 20]
# width = 300

##### EDIT number of sims run in make_inputs.py
start = 0
nsim = 10

slr = [0]
SLR = list(map(lambda x: str(x), slr))

# This inputs the previously assigned random variables
fun = np.loadtxt(indir + 'fun.csv', delimiter=",")
C_FC = np.loadtxt(indir + 'cfc.csv', delimiter=",")
gmm = np.loadtxt(indir + 'gmm.csv', delimiter=",")
wd_var = np.loadtxt(indir + 'wd.csv', delimiter=",")

fs_data, qc_data = get_sgems_sims(soildir, nsim,start = start)
points = get_points(fs_data, qc_data, nsim, geoplot, start = start)
points = interp_elev(points, geotiff_patha, geotiff_pathb) # doesn't need sim/start
store = setup_soil(points, nsim, start = start)
wd_base = get_wd(datadir, points) # doesn't need sim/start
store = vary_wd(wd_base, wd_var, store, start = start, nsim = nsim)

#### This will save the data at each location so it can be loaded in liq_run.py
with open(outdir + 'store.pkl','wb') as fp:
    pickle.dump(store,fp)


