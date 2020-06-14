# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:18:42 2020

@author: Johannes
"""

import os
from rasterio.merge import merge
from rasterio import open
from rasterio.features import rasterize
import rasterio.mask 
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import pyproj
import geopandas
import matplotlib.pyplot as plt

def directory_structure(path):
    
    """Get the directories subdirectories and tif of the main folder of the
    platetScope data.
    
    Parameters
    ----------------
    
    path: String to the main folder
    
    Returns
    ----------------
    Dictionary with path to image files for each day
    and a list of the days present in the folder
    
    """
    
    subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]
    days = [os.path.basename(f) for f in subdirectories]
    pathDays = dict(zip(days, [""] *len(days)))
    
    
    for j in range(len(subdirectories)):
        tifs= []
        for (dirpath, dirnames, filenames) in os.walk(subdirectories[j]):
            for x in range(len(filenames)):
                
                if filenames[x].endswith("AnalyticMS_SR.tif"):
                    tifs.append(os.path.join(dirpath + "\\",filenames[x]))


        pathDays[days[j]] = tifs
    
    return pathDays, days


def merge_tifs(images, outpath):
    
    """create mosaic with rasterio of the images taken at each day
    
    Parameters
    ----------------
    
    images: list of strings to the images' locations
    outpath: path to save mosaic to
    
    Returns
    ----------------
    
    """
    
    src_files_to_mosaic = []
    
    
    # open the images with rasterio
    for fp in images:
        src = open(fp)
        src_files_to_mosaic.append(src)
    crs = src.crs
        
    # merge to a mosaic
    mosaic, out_trans = merge(src_files_to_mosaic, method = 'min')
            
    
    # copy meta data for transform matrix, crs, height, width etc.
    out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": crs
                }
               )
    with open(outpath, "w", **out_meta) as dest:
        dest.write(mosaic)
        
        
def clip_to_StudyArea(image, shapefile, outpath):
        
    """Clips a given image with the given shapefile.
        Parameters
    ----------------
    
    image: String to the images' locations
    outpath: path to save clipped image to
    
    Returns
    ----------------
    """
    # open the shapefile and the image
    data = gpd.read_file(shapefile)
    for feature in data:
        if feature == "geometry":
            clip = data[feature]
            
    crsShape = str(data.crs)
    crsShape = crsShape.lower()
    
    img = rasterio.open(image)
    
    crsImage = str(img.crs)
    crsImage = crsImage.lower()
    
    # change crs if required
    if crsImage is not crsShape:
        data = data.to_crs(crs = crsImage)
    
    out_image, out_transform = rasterio.mask.mask(img, clip, crop = True)
    out_meta = img.meta
        
    crsImage = str(img.crs)
    crsImage = crsImage.lower()
    
    # change crs if required
    if crsImage is not crsShape:
        data = data.to_crs(crs = crsImage)
        
    out_meta.update({"driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform})

    with rasterio.open(outpath, "w", **out_meta) as dest:
        dest.write(out_image)    
        
        
def ndvi_PlanetScope(image, outpath):
    
    """Computes the NDVI for a given image and saves it to a location
        
    
    Parameters
    ----------------
    
    image: string to the images locations
    outpath: path to save mosaic to
    
    Returns
    ----------------   
    """
    
    with rasterio.open(image) as img:
        red = img.read(3)
        nir = img.read(4)
        out_meta = img.profile
        out_meta.update(count = 1, dtype = np.float64
            )
        
    # Convert the red and nir bands to numpy arrays with type float for 
    # NDVI calculation
    red = np.array(red, dtype=float)
    nir = np.array(nir, dtype=float)
    
    # Create variables for the NDVI formula
    num = nir - red
    denom = (nir + red) + 0.00000000001  
    
    # we add the 0.00000000001, so that we never divide by 0.0
    
    # Create a new numpy array by dividing num and denom
    ndvi = np.divide(num,denom)
    
    with open(outpath, "w", **out_meta) as dest:
        dest.write(ndvi.astype(np.float64), indexes = 1)
        
def extract_values(image, shapefile):
    
    """Extracts pixel values of a given images inside of a shapefile.
    
    Parameters
    ----------------
    image: Path to image
    shapefile: Path to shapefile
    
    Return
    ----------------
    np.ndarray with the values of the pixels
    """
    
    shapefile = gpd.read_file(shapefile)
    image = rasterio.open(image, 'r')
    
    image_crs = str(image.crs).lower()
    shapefile_crs = shapefile.crs
    shapefile_crs = str(shapefile_crs)
    shape = image.shape
    T = np.array(image.transform).reshape(3,3)
    
    # reproject shapefile to same EPSG as image
    if image_crs != shapefile_crs:
        shapefile = shapefile.to_crs({'init': image_crs})
        
    mask = rasterize(
        zip(*[shapefile.geometry, np.ones(len(shapefile.geometry))]),
        out_shape=shape,
        transform=(T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2]),
        dtype=np.uint16
    ) > 0
    image_pixels = image.read()
    values = image_pixels[0,mask]
    
    return values 
        
def image_stacking(images, outpath, meta=None, out_shape=None,dtype=np.float64):
    """Makes a layer stackout of the provided images or arrays.
    
    Parameters
    ----------------
    images : list of image paths or list of np.ndarrays
    outpath: path to output location
    meta: optional, if input are arrays. Metadata has to have the form of 
          rasterio format.
    out_shape: Shape of the image/array 
    dtype: data type of the array
    
    Examples for arrays as input
    ----------
    Create a 3-band input image
    >>> im_shape = (3, 1024, 1280)
    >>> dtype = np.uint16
    >>> im = np.zeros(im_shape, dtype = dtype)
    
    Write the required metadata for the image
    >>> meta_d = {'driver': 'GTiff',
    ...           'dtype': 'uint16',
    ...           'nodata': None,
    ...           'width': 1280,
    ...           'height': 1024,
    ...           'count': 3,
    ...           'crs': None,
    ...           'transform': Affine(1.0, 0.0, 1.0,
    ...                               0.0, -1.0, 1.0)
    ...          }
    
    Provide output path
    >>> outpath = "D:\Files\image.tif"
    
    Write the image
    >>> write_image(outpath, meta_d, im, dtype)
    """
  
    arr = np.zeros(out_shape, dtype = dtype)
    n_bands = len(images)
    for i, img in enumerate(images):
        raster = rasterio.open(img,mode="r")
        meta = raster.meta
        arr[i, :, :] = raster.read()
    meta.update({'count': n_bands,
                 'dtype': dtype})
    with rasterio.open(outpath, "w",**meta) as dst:
        dst.write(arr)
        
if __name__ == "__main__":
    """
    
    path = r"F:\planet\Planet_Daten"
    
    images = directory_structure(path)
    tifs = images[0]
    days = images[1]
    outPath = r"F:\planet\Planet_Daten"
    #for x in range(len(tifs)):
    #    merge_tifs(tifs[days[x]],os.path.join(outPath +"\\" + days[x] + 
    #                                         "mosaic.tif" ))
    
    mosaics = [os.path.join(path, f) for f in os.listdir(path)
               if f.endswith("mosaic.tif")]
    #shapefile = r"F:\\planet\\shapefiles\\Planet_TestSite_UG.shp"
    
    clipped = [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith("clipped.tif")]
    ndvi = [os.path.join(path, f) for f in os.listdir(path)
             if f.endswith("ndvi.tif")]
    #for file in range(len(mosaics)):
        
    #    clip_to_StudyArea(mosaics[file],shapefile, 
    #                  os.path.join(outPath +"\\" + days[file] + 
    #                                         "clipped.tif"))
        
    #for file in range(len(clipped)):
    #    
    #    ndvi_PlanetScope(clipped[file], 
    #                  os.path.join(outPath +"\\" + days[file] + 
    #                                         "ndvi.tif"))
    #x=rasterio.open(ndvi[0]).shape

    #image_stacking(ndvi, os.path.join(outPath +"\\ndviStack.tif"),
    #               out_shape= (len(ndvi),x[0],x[1]))
    shapefile_broadleaf = r"F:\planet\shapefiles\laubwald.shp"
    shapefile_coniferous =  r"F:\planet\shapefiles\nadelwald.shp"
    shapefile_agriculture = r"F:\planet\shapefiles\landwirtschaft.shp"
    ndvi_broadleaf = []
    ndvi_coniferous = []
    ndvi_agriculture = []
    
    for file in ndvi:
        print(file)
        values = extract_values(file, shapefile_broadleaf)
        total_sum = values.sum()
        length_values = np.count_nonzero(~np.isnan(values))
        average = total_sum/length_values
        ndvi_broadleaf.append(average)

    for file in ndvi:
        values = extract_values(file, shapefile_coniferous)
        total_sum = values.sum()
        length_values = np.count_nonzero(~np.isnan(values))
        average = total_sum/length_values
        ndvi_coniferous.append(average)
    
    for file in ndvi:
        values = extract_values(file, shapefile_agriculture)
        total_sum = values.sum()
        length_values = np.count_nonzero(~np.isnan(values))
        average = total_sum/length_values
        ndvi_agriculture.append(average)
    
    ndvidays=[]
    for i,day in enumerate(days):
        for j,path in enumerate(ndvi):
            if day in os.path.basename(path):
                ndvidays.append(int(day))
           
    
    plt.plot(ndvidays, ndvi_broadleaf)
    plt.axis([0, 1,0,21])
    plt.show()  
    """
    
  
    
