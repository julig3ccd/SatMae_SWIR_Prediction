import rasterio
from rasterio.plot import show, reshape_as_image, reshape_as_raster
import numpy as np
#import matplotlib




def open_image(img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)
        return img  # (h, w, c)


def main():
      imgAsRaster=reshape_as_raster(open_image("TestData/airport_0_1.tif"))
      img=reshape_as_image(open_image("TestData/airport_0_1.tif"))
      #matplotlib.pyplot.imshow(img)
      #show(imgAsRaster)
      
      print("image: ", img)
      img[:, :,11:13 ] = 0
      print("image: ", img[:,:,8:13])
      
      


main()
