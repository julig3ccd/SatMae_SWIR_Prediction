import rasterio
import numpy as np
import rasterio.plot




def open_image(img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)


def main():
      img=open_image("TestData/airport_0_1.tif")
      print("image: ", img)
      img[:, :,11:13 ] = 0
      print("image: ", img[:,:,8:13])
      
      


main()
