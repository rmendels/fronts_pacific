from Canny1 import *
from cartopy import crs
import cmocean
import cv2
import numpy as np
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from PIL import Image
import xarray as xr

def isleap(year):
    from datetime import date, datetime, timedelta
    try:
        date(year,2,29)
        return True
    except ValueError: return False


def extract_mur(file_name, file_base = '/Users/rmendels/WorkFiles/fronts/',  lat_min = 22., lat_max = 51., lon_min = -135.,  lon_max = -105.):
    import numpy as np
    import numpy.ma as ma
    from netCDF4 import Dataset
    nc_file = file_base + file_name
    root = Dataset(nc_file)
    lat = root.variables['lat'][:]
    lon = root.variables['lon'][:]
    lat_min_index = np.argwhere(lat == lat_min)
    lat_min_index = lat_min_index[0, 0]
    lat_max_index = np.argwhere(lat == lat_max)
    lat_max_index = lat_max_index[0, 0]
    lon_min_index = np.argwhere(lon == lon_min)
    lon_min_index = lon_min_index[0, 0]
    lon_max_index = np.argwhere(lon == lon_max)
    lon_max_index = lon_max_index[0, 0]
    lon_mur = lon[lon_min_index:lon_max_index + 1]
    lat_mur = lat[lat_min_index:lat_max_index + 1]
    sst_mur = root.variables['analysed_sst'][0, lat_min_index:lat_max_index + 1, lon_min_index:lon_max_index + 1 ]
    sst_mur = np.squeeze(sst_mur)
    sst_mur = sst_mur - 273.15
    root.close()
    return sst_mur, lon_mur, lat_mur


def myCanny(myData, myMask, sigma = 10., lower = .8, upper = .9, use_quantiles = True):
    # because of the way masks operate,  if you read in sst using netcdf4,  then the mask to use is ~sst.mask
    edges, x_gradient, y_gradient, magnitude = canny(myData, sigma = sigma, mask = myMask, low_threshold = lower, high_threshold = upper,
                              use_quantiles = use_quantiles)
    x_gradient = ma.array(x_gradient, mask = myData.mask)
    y_gradient = ma.array(y_gradient, mask = myData.mask)
    magnitude = ma.array(magnitude, mask = myData.mask)
    return edges, x_gradient, y_gradient, magnitude

def my_contours(edges):
    edge_image = edges.astype(np.uint8)
    contours, hierarchy = cv2.findContours(edge_image ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return(contours)

def contours_to_edges(contours, edge_shape, min_len = 10):
    num_contours  = len(contours)
    contour_lens = []
    contour_edges = np.zeros(edge_shape)
    for i in list(range(0, num_contours)):
        contour = contours[i]
        contour_len = contour.shape[0]
        contour_lens.append(contour_len)
        if (contour_len > min_len):
            for ilen in list(range(0, contour_len)):
                xloc = contour[ilen, 0, 1]
                yloc = contour[ilen, 0, 0]
                contour_edges[xloc, yloc] = 1
    return contour_edges, contour_lens

def plot_canny_edges(myData, edges, latitudes, longitudes, title = ' ', fig_size = ([8, 6]) ):
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask = (edges1 == 0))
    edges1_xr = xr.DataArray(edges1, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'edge')
    im1 = myData_xr.plot(cmap = cmocean.cm.thermal)
    im2 = edges1_xr.plot(cmap = plt.cm.gray)
    plt.title(title)
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()


def plot_canny_gradient(my_grad, edges, latitudes, longitudes, title = ' ', fig_size = ([10, 8]) ):
    #edges1 = edges.astype(int)
    #edges1 = ma.array(edges1, mask = (edges1 == 0))
    #edges1_xr = xr.DataArray(edges1, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'edge')
    fig, axes = plt.subplots(ncols=2)
    myData_xr = xr.DataArray(my_grad, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'gradient')
    if(my_grad.min() < 0.):
        myData_xr.plot(cmap = cmocean.cm.balance, ax=axes[0])
    else:
        myData_xr.plot(cmap = cmocean.cm.amp, ax=axes[0])
    #edges1_xr.plot(cmap = plt.cm.gray, ax=axes[0])
    myData_xr = xr.DataArray(np.abs(my_grad), coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'gradient')
    myData_xr.plot.hist(bins = 100, histtype='step', density = True, stacked = True, cumulative=True, ax=axes[1])
    plt.title('')
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()
    fig.suptitle(title, y =  1.0)


def plot_canny_contours(myData, edges, contour_lens, latitudes, longitudes, title = ' ', fig_size = ([8, 6]) ):
    fig, axes = plt.subplots(ncols=2)
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask = (edges1 == 0))
    edges1_xr = xr.DataArray(edges1, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name = 'edge')
    im1 = myData_xr.plot(cmap = cmocean.cm.thermal, ax=axes[0])
    im2 = edges1_xr.plot(cmap = plt.cm.gray, ax=axes[0])
    plt.hist(contour_lens, bins = [1, 5 , 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], histtype='bar', density = False)
    #plt.title('')
    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()
    fig.suptitle(title, y =  1.0)



from numba import jit, float32, float64, int32
@jit(int32[:,:](float32[:], float32[:], float64[:, :]), parallel = True)
def filt5(lon , lat, ingrid):
    ## assume ingrid is ma.array with missing data masked
    l1 = lat.shape[0]
    l2 = lon.shape[0]
    outgrid = np.zeros((l1, l2), np.int32)
    for i in list(range(2, l1 - 2)):
        for j in list(range(2, l2 -2)):
            subg = ingrid[(i - 2):(i + 3),(j - 2):(j + 3)]
            if (np.sum(subg.mask == True) == 25):
               outgrid[i,j] = 0
            else:
                my_max = np.argmax(subg)
                my_min = np.argmin(subg)
                if ((my_max == 12) or (my_min == 12)):
                    outgrid[i, j] =  1
                else:
                    outgrid[i, j] = 0

    outgrid = ma.array(outgrid, mask = ingrid.mask)
    return(outgrid)

@jit(float64[:,:](float32[:], float32[:], float64[:, :], int32[:, :]), parallel = True)
def filt35(lon , lat, ingrid, grid5):
    ## assume ingrid is ma.array with missing data masked
    l1 = lat.shape[0]
    l2 = lon.shape[0]
    outgrid = np.zeros((l1, l2))
    for i in list(range(2, l1 - 2)):
        for j in list(range(2, l2 -2)):
            if ((grid5[i, j] == 0)):
                subg = ingrid[(i - 1):(i + 2),(j - 1):(j + 2)]
                if (np.sum(subg.mask == True) == 9):
                    outgrid[i, j] = ingrid[i, j]
                else:
                    my_max = np.argmax(subg)
                    my_min = np.argmin(subg)
                    if ((my_max == 4) or (my_min == 4)):
                        outgrid[i, j] = ma.median(subg)  # apply median filter if there is peak 3
                    else:
                        outgrid[i, j] = ingrid[i, j]
            else:
                outgrid[i, j] = ingrid[i, j]


    outgrid = ma.array(outgrid, mask = ingrid.mask)
    outgrid[outgrid == 0] = ma.masked
    return(outgrid)

def create_canny_nc(file_year, file_month, file_day, base_dir, lat_min = 22., lat_max = 51.,  lon_min = -135., lon_max = -105.):
    from netCDF4 import Dataset, num2date, date2num
    import numpy as np
    import numpy.ma as ma
    c_file_year = str(file_year)
    c_file_month = str(file_month).rjust(2,'0')
    c_file_day = str(file_day).rjust(2,'0')
    file_name = base_dir + 'Canny_Front_' + c_file_year + c_file_month + c_file_day +  '.nc'
    ncfile  = Dataset(file_name, 'w', format = 'NETCDF4')
    lat_diff = lat_max - lat_min
    latsdim = (lat_diff * 100) + 1
    lats = lat_min + (np.arange(0, latsdim) * 0.01)
    lon_diff = lon_max - lon_min
    lonsdim = (lon_diff * 100) + 1
    lons = lon_min + (np.arange(0, lonsdim) * 0.01)
    #Create Dimensions
    timedim = ncfile.createDimension('time', None)
    latdim = ncfile.createDimension('lat', latsdim)
    londim = ncfile.createDimension('lon', lonsdim)
    altdim = ncfile.createDimension('altitude', 1)
    #Create Variables
    LatLon_Projection = ncfile.createVariable('LatLon_Projection', 'i4')
    time = ncfile.createVariable('time', 'f8', ('time'), zlib = True, complevel = 2)
    altitude = ncfile.createVariable('altitude', 'f4', ('altitude'))
    latitude = ncfile.createVariable('lat', 'f4', ('lat'), zlib = True, complevel = 2)
    longitude = ncfile.createVariable('lon', 'f4', ('lon'), zlib = True, complevel = 2)
    edges = ncfile.createVariable('edges', 'f4', ('time', 'altitude', 'lat', 'lon'), fill_value = -9999.0, zlib = True, complevel = 2)
    x_gradient = ncfile.createVariable('x_gradient', 'f4', ('time', 'altitude', 'lat', 'lon'), fill_value = -9999.0, zlib = True, complevel = 2)
    y_gradient = ncfile.createVariable('y_gradient', 'f4', ('time', 'altitude', 'lat', 'lon'), fill_value = -9999.0, zlib = True, complevel = 2)
    magnitude_gradient = ncfile.createVariable('magnitude_gradient', 'f4', ('time', 'altitude', 'lat', 'lon'), fill_value = -9999.0, zlib = True, complevel = 2)
    # int LatLon_Projection ;
    LatLon_Projection.grid_mapping_name = "latitude_longitude"
    LatLon_Projection.earth_radius = 6367470.
    #float lat(lat) ;
    latitude._CoordinateAxisType = "Lat"
    junk = (lat_min, lat_max)
    latitude.actual_range = junk
    latitude.axis = "Y"
    latitude.grid_mapping = "Equidistant Cylindrical"
    latitude.ioos_category = "Location"
    latitude.long_name = "Latitude"
    latitude.reference_datum = "geographical coordinates, WGS84 projection"
    latitude.standard_name = "latitude"
    latitude.units = "degrees_north"
    latitude.valid_max = lat_max
    latitude.valid_min = lat_min
    #float lon(lon) ;
    longitude._CoordinateAxisType = "Lon"
    junk = (lon_min, lon_max)
    longitude.actual_range = junk
    longitude.axis = "X"
    longitude.grid_mapping = "Equidistant Cylindrical"
    longitude.ioos_category = "Location"
    longitude.long_name = "Longitude"
    longitude.reference_datum = "geographical coordinates, WGS84 projection"
    longitude.standard_name = "longitude"
    longitude.units = "degrees_east"
    longitude.valid_max = lon_max
    longitude.valid_min = lon_min
    #float altitude(altitude) ;
    altitude.units = "m"
    altitude.long_name = "Specified height level above ground"
    altitude.standard_name = "altitude"
    altitude.positive = "up"
    altitude.axis = "Z"
    #double time(time) ;
    time._CoordinateAxisType = "Time"
    junk = ()
    time.actual_range = junk
    time.axis = "T"
    time.calendar = "Gregorian"
    time.ioos_category = "Time"
    time.long_name = "Time"
    time.units = "Hour since 1970-01-01T00:00:00Z"
    time.standard_name = "time"
    #float edges(time, altitude, lat, lon) ;
    edges.long_name = "Frontal Edge"
    edges.missing_value = -9999.
    edges.grid_mapping = "LatLon_Projection"
    edges.coordinates = "time altitude lat lon "
    #float x_gradient(time, altitude, lat, lon) ;
    x_gradient.long_name = "East-West Gradient of SST"
    x_gradient.missing_value = -9999.
    x_gradient.grid_mapping = "LatLon_Projection"
    x_gradient.coordinates = "time altitude lat lon "
    # float y_gradient(time, altitude, lat, lon) ;
    y_gradient.long_name = "North-South Gradient of SST"
    y_gradient.missing_value = -9999.
    y_gradient.grid_mapping = "LatLon_Projection"
    y_gradient.coordinates = "time altitude lat lon "
    # float magnitude(time, altitude, lat, lon) ;
    magnitude_gradient.long_name = "Magnitude of SST Gradient"
    magnitude_gradient.missing_value = -9999.
    magnitude_gradient.grid_mapping = "LatLon_Projection"
    magnitude_gradient.coordinates = "time altitude lat lon "
    ## global
    ncfile.title = "Daily estimated MUR SST Frontal edges, x_gradient, y_gradient and gradient magnitude"
    ncfile.cdm_data_type = "Grid"
    ncfile.Conventions = "COARDS, CF-1.6, ACDD-1.3"
    ncfile.standard_name_vocabulary = "CF Standard Name Table v55"
    ncfile.creator_email = "erd.data@noaa.gov"
    ncfile.creator_name =  "NOAA NMFS SWFSC ERD"
    ncfile.creator_type =  "institution"
    ncfile.creator_url  = "https://www.pfeg.noaa.gov"
    ncfile.Easternmost_Easting = lon_max
    ncfile.Northernmost_Northing = lat_max
    ncfile.Westernmost_Easting = lon_min
    ncfile.Southernmost_Northing =  lat_max
    ncfile.geospatial_lat_max = lat_max
    ncfile.geospatial_lat_min =  lat_min
    ncfile.geospatial_lat_resolution = 0.01
    ncfile.geospatial_lat_units = "degrees_north"
    ncfile.geospatial_lon_max = lon_max
    ncfile.geospatial_lon_min = lon_min
    ncfile.geospatial_lon_resolution = 0.01
    ncfile.geospatial_lon_units = "degrees_east"
    ncfile.infoUrl = ""
    ncfile.institution = "NOAA ERD"
    ncfile.keywords = ""
    ncfile.keywords_vocabulary = "GCMD Science Keywords"
    ncfile.summary = '''Front Edges estimated from daily MUR SST files
    using the Python scikit-image canny algorithm  with sigma = 10., and
    threshold values of .8 and .9,  as well as the OpenCV algorithm findContours.
    The SST x-gradient, y-gradient and gradient magnitude are also included
    '''
    ncfile.license = '''The data may be used and redistributed for free but is not intended
    for legal use, since it may contain inaccuracies. Neither the data
    Contributor, ERD, NOAA, nor the United States Government, nor any
    of their employees or contractors, makes any warranty, express or
    implied, including warranties of merchantability and fitness for a
    particular purpose, or assumes any legal liability for the accuracy,
    completeness, or usefulness, of this information.
    '''
    file_name = c_file_year + c_file_month + c_file_day + '090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
    history = 'created from MUR SST file ' + file_name + 'using python scikit-image canny algorithm, sigma = 10, thresholds of 0.8, 0.9 and OpenCV findContours function'
    ncfile.history = history
    altitude[0] = 0.
    longitude[:] = lons[:]
    latitude[:] = lats[:]
    ncfile.close()






