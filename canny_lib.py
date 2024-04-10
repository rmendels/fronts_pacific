import cmocean
from Canny2 import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from skimage.feature import canny
import xarray as xr

def isleap(year):
    from datetime import date, datetime, timedelta
    try:
        date(year,2,29)
        return True
    except ValueError: return False


def extract_mur(file_name, file_base = '/Users/rmendels/WorkFiles/fronts_pacific/',  lat_min = 22., lat_max = 51., lon_min = -135.,  lon_max = -105.):
    """Extracts sea surface temperature (SST) data from a MUR (Multi-scale Ultra-high Resolution) NetCDF file.

    This function reads a specific geographical subset of SST data from a MUR JPL NetCDF file,
    converting temperatures from Kelvin to Celsius. It focuses on a predefined area by
    latitude and longitude boundaries.

    Args:
        file_name (str): The name of the NetCDF file to process.
        file_base (str): The base directory path where the NetCDF file is located.
            Defaults to '/u00/satellite/front_atlantic/fronts_atlantic/'.
        lat_min (float): The minimum latitude of the geographical area of interest. Defaults to 20.0.
        lat_max (float): The maximum latitude of the geographical area of interest. Defaults to 50.0.
        lon_min (float): The minimum longitude of the geographical area of interest. Defaults to -90.0.
        lon_max (float): The maximum longitude of the geographical area of interest. Defaults to -60.0.

    Returns:
        sst_mur (numpy.ndarray): The sea surface temperature data in Celsius for the specified region.
        lon_mur (numpy.ndarray): The array of longitude values within the specified region.
        lat_mur (numpy.ndarray): The array of latitude values within the specified region.

    Example:
        >>> sst_data, lon_values, lat_values = extract_mur('mur_sst_file.nc')
    """
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
    """Applies the Canny edge detection algorithm to input data.

    Uses the Canny edge detection algorithm on the provided data array, utilizing the dataset mask.
    The function configures the algorithm's sensitivity through the sigma, lower, and upper threshold parameters.

    Args:
        myData (ndarray): Input data for edge detection.
        myMask (ndarray): Boolean mask for the data, where True indicates a valid data point.
        sigma (float): Standard deviation of the Gaussian filter. Defaults to 12.5.
        lower (float): Lower bound for hysteresis thresholding. Defaults to 0.8.
        upper (float): Upper bound for hysteresis thresholding. Defaults to 0.9.
        use_quantiles (bool): Whether to use quantiles for thresholding. Defaults to True.

    Returns:
        x_gradient (MaskedArray): The gradient of the data in the x-direction, masked similarly to input data.
        y_gradient (MaskedArray): The gradient of the data in the y-direction, masked similarly to input data.
        magnitude (MaskedArray): The magnitude of the gradient, masked similarly to input data.

    Example:
        >>> x_grad, y_grad, magnitude = myCanny(data, ~data.mask)
    """
    # because of the way masks operate,  if you read in sst using netcdf4,  then the mask to use is ~sst.mask
    y_gradient, x_gradient, magnitude  = canny2(myData, sigma = sigma, mask = myMask, low_threshold = lower, high_threshold = upper,
                              use_quantiles = use_quantiles)
    edges = canny(myData, sigma = sigma, mask = myMask, low_threshold = lower, high_threshold = upper,
                              use_quantiles = use_quantiles)
    x_gradient = ma.array(x_gradient, mask = myData.mask)
    y_gradient = ma.array(y_gradient, mask = myData.mask)
    magnitude = ma.array(magnitude, mask = myData.mask)
    return edges, x_gradient, y_gradient, magnitude

def my_contours(edges):
    """Finds contours in an edge-detected image.

    Uses OpenCV's findContours function to detect contours in a binary edge-detected image.

    Args:
        edges (ndarray): Binary edge-detected image where edges are marked as True or 1.

    Returns:
        contours (list): A list of contours found in the image, where each contour is represented as an array of points.

    Note:
        Requires OpenCV (cv2) for contour detection. Ensure cv2 is installed and imported as needed.

    Example:
        >>> contours = my_contours(edge_detected_image)
    """
    edge_image = edges.astype(np.uint8)
    contours, hierarchy = cv2.findContours(edge_image ,cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    return(contours)

def contours_to_edges(contours, edge_shape, min_len = 20):
    """Converts contour points into a binary edge image.

    This function iterates through a list of contours and marks corresponding points on a binary
    edge image. Only contours longer than a specified minimum length are processed to filter out
    smaller, potentially less significant features.

    Args:
        contours (list): A list of contour arrays, where each contour is represented by its points.
        edge_shape (tuple): The shape of the output edge image (height, width).
        min_len (int): Minimum length of a contour to be included in the edge image. Defaults to 10.

    Returns:
        contour_edges (numpy.ndarray): A binary edge image with marked contours.
        contour_lens (list): Lengths of all contours found, for further analysis.

    Example:
        >>> edges, lengths = contours_to_edges(contours, image.shape)
    """
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

def plot_canny_contours3(myData, edges, contour_lens, latitudes, longitudes, title=' ', dot_size=.5, fig_size=([8, 6])):
    """Plots Canny edge detection results overlaid on the original data.

    Uses xarray and matplotlib to display the original data and its Canny edges. The edges are plotted
    in black on top of the original data colored by a thermal colormap.

    Args:
        myData (numpy.ndarray): Original data array.
        edges (numpy.ndarray): Binary edge data from Canny edge detection.
        latitudes (numpy.ndarray): Latitude coordinates for the data.
        longitudes (numpy.ndarray): Longitude coordinates for the data.
        title (str): Title for the plot. Defaults to a blank space.
        fig_size (list): Figure size. Defaults to [8, 6].

    Example:
        >>> plot_canny_edges(data, edges, lats, lons, 'Canny Edge Detection', [10, 8])
    """
    plt.rcParams["figure.figsize"] = fig_size
    fig, axes = plt.subplots(ncols=2, figsize=fig_size)
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name='sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    # Plotting myData_xr with a colormap
    im1 = myData_xr.plot(cmap=cmocean.cm.thermal, ax=axes[0], add_colorbar=True)
    # Ensure im1 is fully rendered
    plt.draw()
    # Converting edges to masked array to overlay
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask=(edges1 == 0))
    # Find indices where edges exist
    edge_indices = np.where(edges1 == 1)
    # Convert indices to coordinates
    edge_lat_coords = latitudes[edge_indices[0]]
    edge_lon_coords = longitudes[edge_indices[1]]
    # Overlay edges as black dots on the first subplot with higher zorder
    axes[0].scatter(edge_lon_coords, edge_lat_coords, color='black', s=dot_size, zorder=3)
    # Plot histogram on the second subplot
    axes[1].hist(contour_lens, bins=[1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], histtype='bar', density=False)
    plt.tight_layout()
    fig.suptitle(title, y=1.0)


def plot_canny_contours2(myData, edges, contour_lens, latitudes, longitudes, title=' ', dot_size = 10, fig_size=([8, 6])):
    """Plots Canny edge detection results overlaid on the original data.

    Uses xarray and matplotlib to display the original data and its Canny edges. The edges are plotted
    in grayscale on top of the original data colored by a thermal colormap.

    Args:
        myData (numpy.ndarray): Original data array.
        edges (numpy.ndarray): Binary edge data from Canny edge detection.
        latitudes (numpy.ndarray): Latitude coordinates for the data.
        longitudes (numpy.ndarray): Longitude coordinates for the data.
        title (str): Title for the plot. Defaults to a blank space.
        fig_size (list): Figure size. Defaults to [8, 6].

    Example:
        >>> plot_canny_edges(data, edges, lats, lons, 'Canny Edge Detection', [10, 8])
    """
    plt.rcParams["figure.figsize"] = fig_size
    fig, axes = plt.subplots(ncols=2, figsize=fig_size)
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name='sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    # Plotting myData_xr with a colormap
    im1 = myData_xr.plot(cmap=cmocean.cm.thermal, ax=axes[0], add_colorbar=True)
    # Converting edges to masked array to overlay
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask=(edges1 == 0))
    # Find indices where edges exist
    edge_indices = np.where(edges1 == 1)
    # Convert indices to coordinates
    edge_lat_coords = latitudes[edge_indices[0]]
    edge_lon_coords = longitudes[edge_indices[1]]
    # Overlay edges as black dots on the first subplot
    axes[0].scatter(edge_lon_coords, edge_lat_coords, color='black', s=dot_size)  # Adjust s as needed
     # Plot histogram on the second subplot
    axes[1].hist(contour_lens, bins=[1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], histtype='bar', density=False)
    plt.tight_layout()
    fig.suptitle(title, y=1.0)

def plot_canny_contours(myData, edges, contour_lens, latitudes, longitudes, title=' ', dot_size=10, fig_size=([8, 6])):
    """Plots Canny edge detection results overlaid on the original data.

    Uses xarray and matplotlib to display the original data and its Canny edges. The edges are plotted
    in grayscale on top of the original data colored by a thermal colormap.

    Args:
        myData (numpy.ndarray): Original data array.
        edges (numpy.ndarray): Binary edge data from Canny edge detection.
        latitudes (numpy.ndarray): Latitude coordinates for the data.
        longitudes (numpy.ndarray): Longitude coordinates for the data.
        title (str): Title for the plot. Defaults to a blank space.
        fig_size (list): Figure size. Defaults to [8, 6].

    Example:
        >>> plot_canny_edges(data, edges, lats, lons, 'Canny Edge Detection', [10, 8])
    """
    plt.rcParams["figure.figsize"] = fig_size
    fig, axes = plt.subplots(ncols=2, figsize=fig_size)
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name='sst')
    myData_xr.values[myData_xr.values < 5.] = 5.
    # Plotting myData_xr with a colormap
    im1 = myData_xr.plot(cmap=cmocean.cm.thermal, ax=axes[0], add_colorbar=True)
    # Ensure im1 is fully rendered
    plt.draw()
    # Converting edges to masked array to overlay
    edges1 = edges.astype(int)
    edges1 = ma.array(edges1, mask=(edges1 == 0))
    # Find indices where edges exist
    edge_indices = np.where(edges1 == 1)
    # Convert indices to coordinates
    edge_lat_coords = latitudes[edge_indices[0]]
    edge_lon_coords = longitudes[edge_indices[1]]
    # Overlay edges as black dots on the first subplot with higher zorder
    axes[0].scatter(edge_lon_coords, edge_lat_coords, color='black', s=dot_size, zorder=3)
    # Plot histogram on the second subplot
    axes[1].hist(contour_lens, bins=[1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], histtype='bar', density=False)
    plt.tight_layout()
    fig.suptitle(title, y=1.0)


def plot_canny_contours_old(myData, edges, contour_lens, latitudes, longitudes, title = ' ', fig_size = ([8, 6]) ):
    """Plots bathymetric (seafloor depth) data.

    This function uses xarray and matplotlib to visualize bathymetric data. The depth values are
    displayed using a colormap designed for deep water.

    Args:
        depth (numpy.ndarray): Array of depth values.
        latitudes (numpy.ndarray): Latitude coordinates for the depth data.
        longitudes (numpy.ndarray): Longitude coordinates for the depth data.
        title (str): Title of the plot. Defaults to a blank space.
        fig_size (list): Dimensions of the plot. Defaults to [10, 8].

    Example:
        >>> plot_bathy(depth_data, lat_array, lon_array, 'Bathymetric Data Visualization')
    """
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

def plot_canny_gradient(my_grad, edges, latitudes, longitudes, title = ' ', fig_size = ([10, 8]) ):
    """Plots the gradient magnitude from Canny edge detection alongside its histogram.

    Visualizes the gradient magnitude as an image and its distribution as a histogram in a side-by-side view.
    Uses xarray for plotting the gradient and matplotlib for histograms.

    Args:
        my_grad (numpy.ndarray): Gradient magnitude array.
        latitudes (numpy.ndarray): Latitude coordinates for the gradient data.
        longitudes (numpy.ndarray): Longitude coordinates for the gradient data.
        title (str): Title for the subplot. Defaults to a blank space.
        fig_size (list): Figure dimensions. Defaults to [10, 8].

    Example:
        >>> plot_canny_gradient(gradient_magnitude, latitudes, longitudes, 'Gradient and Histogram')
    """
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
    """Plots bathymetric (seafloor depth) data.

    This function uses xarray and matplotlib to visualize bathymetric data. The depth values are
    displayed using a colormap designed for deep water.

    Args:
        depth (numpy.ndarray): Array of depth values.
        latitudes (numpy.ndarray): Latitude coordinates for the depth data.
        longitudes (numpy.ndarray): Longitude coordinates for the depth data.
        title (str): Title of the plot. Defaults to a blank space.
        fig_size (list): Dimensions of the plot. Defaults to [10, 8].

    Example:
        >>> plot_bathy(depth_data, lat_array, lon_array, 'Bathymetric Data Visualization')
    """
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


def plot_canny_edges(myData, edges, latitudes, longitudes, title=' ', fig_size=(8, 6), dot_size=10):
    # Adjust figure size at the beginning
    plt.figure(figsize=fig_size)

    # Create DataArray for myData
    myData_xr = xr.DataArray(myData, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name='sst')
    myData_xr.values[myData_xr.values < 5.] = 5.

    # Create a masked array for edges where zeros are masked
    edges1 = edges.astype(bool)

    # Create DataArray for edges
    edges1_xr = xr.DataArray(edges1, coords=[latitudes, longitudes], dims=['latitude', 'longitude'], name='edge')

    # Plot myData
    myData_xr.plot(cmap=cmocean.cm.thermal)

    # Extract the indices of edges1 where there are edges (non-zero)
    y, x = np.where(edges1)

    # Convert indices to latitude and longitude using the coords arrays
    lat_edges = latitudes[y]
    lon_edges = longitudes[x]

    # Plot edges using scatter to control dot size and color
    plt.scatter(lon_edges, lat_edges, color='black', s=dot_size)

    # Set title and layout
    plt.title(title)
    plt.tight_layout()





#def create_canny_nc(file_year, file_month, file_day, base_dir = '/u00/satellite/front/', lat_min = 22., lat_max = 51.,  lon_min = -135., lon_max = -105.):
def create_canny_nc(file_year, file_month, file_day, base_dir, lat_min = 22., lat_max = 51.,  lon_min = -135., lon_max = -105.):
    """Creates a NetCDF file to store Canny edge detection results on sea surface temperature (SST) data.

    This function generates a NetCDF file containing the results of Canny edge detection applied to SST data,
    including the detected edges, and the gradients and magnitude of gradients of SST. The data covers a specific
    geographical region and a specific date, as defined by the input parameters.

    Args:
        file_year (int): Year of the SST data to process.
        file_month (int): Month of the SST data to process.
        file_day (int): Day of the SST data to process.
        base_dir (str): The directory where the NetCDF file will be saved. Defaults to '/PFELData2/front_atlantic/'.
        lat_min (float): The minimum latitude of the geographical region of interest. Defaults to 25.0.
        lat_max (float): The maximum latitude of the geographical region of interest. Defaults to 45.0.
        lon_min (float): The minimum longitude of the geographical region of interest. Defaults to -90.0.
        lon_max (float): The maximum longitude of the geographical region of interest. Defaults to -70.0.

    Returns:
        str: The path of the created NetCDF file.

    Example:
        >>> nc_path = create_canny_nc(2020, 7, 15)
        >>> print(f"NetCDF file created at: {nc_path}")
    """
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






