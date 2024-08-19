import numpy as np
import matplotlib.pyplot as plt

def create_swir_img_from_tensor(image, num_channels): 
    image = image.cpu().numpy()
    npimgtransposed = np.transpose(image, (1, 2, 0))
    stacked_image = np.zeros((npimgtransposed.shape[0], npimgtransposed.shape[1], 3))
    
    if num_channels == 2:
        # Assign the red channel to the first channel
        np.copyto(stacked_image[:, :, 0], npimgtransposed[:,:,0])
        # Assign the green channel to the second channel
        np.copyto(stacked_image[:, :, 1], npimgtransposed[:,:,1])
    elif num_channels == 10:
        # Assign the red channel to the first channel
        np.copyto(stacked_image[:, :, 0], npimgtransposed[:,:,8])
        # Assign the green channel to the second channel
        np.copyto(stacked_image[:, :, 1], npimgtransposed[:,:,9])   

    return stacked_image

def save_comparison_fig_from_tensor(final_images,name,num_channels=2, target_images=None):  # final_image shape: [8,2,96,96]


    
    for idx, img in enumerate(final_images) :
    
        output = create_swir_img_from_tensor(img, num_channels)
        if target_images is not None:
            target = create_swir_img_from_tensor(target_images[idx],num_channels)

        # Display the image using matplotlib
         #print("image shape: ", image_np.shape)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
        ax[0].imshow(output) 
        ax[0].set_title('Output')
        ax[0].axis('off')  # Hide axes

        if target_images is not None:
             # Plot the 'target' image
            ax[1].imshow(target) 
            ax[1].set_title('Target')
            ax[1].axis('off')  # Hide axes

        plt.savefig(f'imgOut/{name}_img_{idx}.png')
        plt.close()
    