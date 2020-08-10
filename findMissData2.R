#####
#
# Created: 2019/12/16
# Author: Jari
#
#####

#Find missing data in images with same dimensions

#input: raster images with potentially different coverage 
#output: raster with count of missing values for each pixel

#needs data with same extent and dimensions

#install.packages("tiff")
library(tiff)
#install.packages("caTools")
library(caTools)
#install.packages("raster")
library(raster)
#install.packages("CRAN")
library(CRAN)
#install.packages("rgdal")
library(rgdal)


setwd("D:\\Geoinformatik\\Planetscope Hunsrück\\Planet\\Planet_Daten")

## list with path of images that will be searched
files <- list.files( pattern = "\\.tif$")
print(files)

#load first image for dimensions and create empty array to fill with count of missing values
one_layer <- raster::as.matrix(raster(files[0]))
miss_data <- array(data = 0, c(dim(one_layer)[1], dim(one_layer)[2]))

#search loop
pb <- winProgressBar(title = "progress bar", min = 0, max = length(files), width = 300)
for (image in 1:length(files)){
  
  # load current image and store in temporary raster object
  tmp_path <- files[image]
  tmp_matrix <- raster::as.matrix(raster(tmp_path))
  
  #search for no data 
  for (a in 1:dim(tmp_matrix)[1]){
    for (b in 1:dim(tmp_matrix)[2]){
      if (is.na(tmp_matrix[a,b])){
        miss_data[a,b] <- miss_data[a,b] + 1
      }
      else if (tmp_matrix[a,b] <= 0){   #only works if negative values or 0 is used to indicate no data
        miss_data[a,b] <- miss_data[a,b] + 1
      }
      
    }
  }
  
  setWinProgressBar(pb, image, title=paste( round(image/length(files)*100, 0),"% done"))
  
}
close(pb)


#create Envi raster image 
write.ENVI(miss_data, "miss_data" ,interleave <- "bsq")



