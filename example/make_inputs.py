## make_inputs.py 
## by Emily Mongold, emongold<at>stanford<dot>edu
## Last updated 9/21/2023
## This script generates the inputs for the liquefaction model.

# imports #
from liquefaction import *
from joblib import Parallel, delayed

### Change filepaths to the input locations ####
geotiff_patha = './inputs/elev/ned03m37122g2.tif'
geotiff_pathb = './inputs/elev/ned03m37122g3.tif'
soildir = './inputs/soil/'
imdir = './inputs/generate_IMs/'
datadir = './inputs/CPT/'
geoplot = './inputs/alameda_plots/Alameda_shape.geojson'

###### Choose where to save the outputs locally  ########
outdir = './outputs/'
os.mkdir(outdir)  # run for the first time to get the directory
# soil grid properties:
# shape = [33, 20, 20]
# width = 300

site_file = imdir + 'Alameda_sites.csv'
run_pypsha(site_file, 2, imdir)

#### Choose the number of simulations, must generate more soil simulations to run over 10 ####
nsim = 10

slr = [0]
SLR = list(map(lambda x: str(x), slr))

# This block creates random inputs of each unknown variable.
C_FC = np.array(random.sample(range(-3000000, 3000000), nsim))/10000000
wd_var = np.array(random.sample(range(-15000000, 15000000), nsim))/10000000
fun = np.array(np.random.choice([0, 1], size=(nsim,)))
gmm = np.array(np.random.choice([0, 1, 2], size=(nsim,)))

# This block saves those inputs for postprocessing
np.savetxt(outdir + 'cfc.csv', C_FC, delimiter=",")
np.savetxt(outdir + 'wd.csv', wd_var, delimiter=",")
np.savetxt(outdir + 'fun.csv', fun, delimiter=",")
np.savetxt(outdir + 'gmm.csv', gmm, delimiter=",")

# This block gets M and pga from pypsha outputs, considering chosen gmm
event_path = imdir + 'event_save.pickle'
M = get_mags_pypsha(event_path)
np.savetxt(outdir + 'M.csv', M, delimiter=",")
run_pypsha_pgas(imdir,gmm)

# This block generates the data gridpoints in a structure that has the soil data
fs_data, qc_data = get_sgems_sims(soildir, nsim)
points = get_points(fs_data, qc_data, nsim, geoplot)

# This plot assigns pgas to each gridpoint based on the inputs.
names = list(range(len(points)))
pgas = get_pgas_from_grid_pypsha(imdir, nsim, names, geoplot, gmm)
np.savetxt(outdir + 'pgas.csv',pgas,delimiter=",")

