## liq_run.py
## by Emily Mongold, emongold<at>stanford<dot>edu
## Last updated 9/21/2023
## This script runs the liquefaction model for the provided example.


from liquefaction import *

outdir = './outputs/'
# soil grid properties:
# shape = [33, 20, 20]
# width = 300

### edit this nsim ###
nsim = 10
slr = [0]
SLR = list(map(lambda x: str(x), slr))

M = np.loadtxt(outdir + 'M.csv')
pgas = pd.read_csv(outdir + 'pgas.csv')
pgas.drop(['Unnamed: 0'],axis=1,inplace=True)
fun = np.loadtxt(outdir + 'fun.csv', delimiter=",")
C_FC = np.loadtxt(outdir + 'cfc.csv', delimiter=",")


#### Edit this to have the correct pkl file #####
with open(outdir + 'store.pkl','rb') as fp:
    store = pickle.load(fp)

##### Edit to the simulations set up in liq_setup.py #####
startsim = 0
stopsim = 10

lpi = Parallel(n_jobs=-1,require = 'sharedmem',max_nbytes = None)(delayed(liqcalc)(i, fun=fun, store=store,
                                          slr=slr, mags=M, pgas=pgas, C_FC=C_FC,SLR=SLR) for i in range(startsim, stopsim))


mini = {}
for i in range(len(lpi)):
    key = list(lpi[i].keys())[0]
    mini[key] = store[key]
gdfp = restruct3(lpi, mini)

save_part(outdir, gdfp, stopsim)




