## run_loss.py
## by Emily Mongold, emongold<at>stanford<dot>edu
## Last updated 9/21/2023
## This script runs the loss analysis for the full outputs provided, or can be run for the example.


from liquefaction import *

bldgs = pd.read_csv('./full_outputs/Pelicun_Alameda_Inventory.txt')
bldPGA = np.load('./full_outputs/bldPGA.npy')

###### Setting up the buildings dataframes #########
bldgs_gdf = gpd.GeoDataFrame(bldgs,geometry=gpd.points_from_xy(bldgs.Longitude,bldgs.Latitude))
bldgs_gdf.set_crs(epsg=4326,inplace=True)
Alameda = gpd.read_file('./inputs/alameda_plots/Alameda_shape.geojson')
bldgs = gpd.sjoin(bldgs_gdf,Alameda)
bldgs.reset_index(inplace=True,drop=True)
bldgs = bldgs.drop(['index_right'], axis=1)

#### Converting the pgas to LRs ########
LR_eq = np.zeros(shape=bldPGA.shape)
st = []
cls = []

for (i, j) in itertools.product(range(len(bldgs)), range(np.shape(bldPGA)[0])):
    cl = ca_cl(bldgs['YearBuilt'][i])
    cls.append(cl)
    STR = type_year(bldgs['OccupancyClass'][i],bldgs['YearBuilt'][i])
    st.append(STR)
    
    beta = 0.64
    theta = get_theta(STR, cl)
    cdf = stats.lognorm(s=beta, scale=theta).cdf(bldPGA[j,i])
    pdf = np.append([1],cdf) - np.append(cdf, [0])
    DS = np.random.choice([0,1,2,3,4], p = pdf)

    LR_eq[j,i] = (loss_from_damage(DS))

np.save('./full_outputs/LRs.npy',LR_eq)
