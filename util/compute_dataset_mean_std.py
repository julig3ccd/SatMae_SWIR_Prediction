import numpy as np
import pandas as pd
import rasterio
import os

def create_df(directory_path):
    fileData = []
    directory_list = os.listdir(directory_path)
    img_id = 0

    for file in directory_list:
        if file.endswith(".tiff"):
            currentFileData= (directory_path+"/"+file, img_id)
            fileData.append(currentFileData)
            img_id += 1

    if len(fileData) != 0:
        df = pd.DataFrame(fileData, columns=['image_path','image_id'])
    else:
        raise ValueError('No tiff files found in directory');  
    return df      

def open_image(img_path, self=None):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)


def compute_channel_means(df):
        
        img_list= [open_image(img_path) for img_path in df['image_path']]
        print("IMAGE LIST LEN: ",len(img_list))
        print("DF LEN: ",len(df))
        channel_mean_df= pd.DataFrame(columns=['channel0','channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8','channel9','channel10','channel11','channel12'])
        for i, img in enumerate(img_list):
            print("IMG SHAPE: ",img.shape)
            imgMeans = []
            for j, channel in enumerate(range(img.shape[2])):
                flat_channel = img[:,:,channel].flatten()
                local_channel_mean = np.mean(flat_channel)
                imgMeans.append(local_channel_mean)
            channel_mean_df.loc[len(channel_mean_df)] = (imgMeans)    

        return channel_mean_df.mean()       

def __main__():
     directory_path = "~/data/train"
     df=create_df(directory_path)
     means=compute_channel_means(df)
     print("Means:",means)
