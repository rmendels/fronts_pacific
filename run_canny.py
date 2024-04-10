# try converting to image first,  then doing th ealgorithm
# Thresholds employed are 0.006 and 0.0015C km1.


from canny_lib import *
from netCDF4 import Dataset
import numpy as np
from scipy.misc import toimage

root = Dataset('/Users/rmendels/WorkFiles/fronts/20171213090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
lats = root.variables['lat'][:]
lons = root.variables['lon'][:]
lat_min = np.where(lats == 21)
lat_min = int(lat_min[0])
lat_max = np.where(lats == 55)
lat_max = int(lat_max[0])
lon_min = np.where(lons == -135.)
lon_min = int(lon_min[0])
lon_max = np.where(lons == -105)
lon_max = int(lon_max[0])
sst = root.variables['analysed_sst'][0, (lat_min -1):lat_max, (lon_min - 1):lon_max]
root.close()
lat_grid = lats[ (lat_min -1):lat_max]
lon_grid = lons[ (lon_min - 1):lon_max]
edges = myCanny(sst)
plot_canny_edges(sst, edges, lat_grid, lon_grid)
sst_med = my_med_filter(sst, med_filter_param = 7)
edges_med = myCanny(sst_med)
plot_canny_edges(sst_med, edges_med, lat_grid, lon_grid)

