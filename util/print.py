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

def get_binary_swir_mask_from_tensor(mask):  # mask shape: [3,144] - [Channel Groups, 12*12 patches] - [:,i] --> has 0 (keep) or 1 (masked)
    mask =mask.cpu().numpy()
    print("MASK SHAPE", mask.shape)
    swir_mask = mask[3,:].reshape(12,12)
    
    return swir_mask

def get_masked_input_img_from_tensor(input,mask):

    swir_input_full=create_swir_img_from_tensor(input,num_channels=10) #shape [96,96,3]
    binary_mask = get_binary_swir_mask_from_tensor(mask)               #shape [12,12]
    
    # resize the binary mask to [96, 96] (nearest-neighbor interpolation)
    resized_mask = np.resize(binary_mask, (96, 96), order=0, preserve_range=True)

    # Apply the resized mask to the image
    # Remember, zeros are 'keep', so no need to invert the mask
    masked_image = swir_input_full * (resized_mask[..., np.newaxis])  # Apply mask across all channels

    return masked_image


def save_comparison_fig_from_tensor(final_swir_images ,name ,num_channels=2, target_images=None, mask=None,input=None ):  # final_image shape: [8,2,96,96]


    
    for idx, img in enumerate(final_swir_images) :
    
        output = create_swir_img_from_tensor(img, num_channels)
        if target_images is not None:
            target = create_swir_img_from_tensor(target_images[idx],num_channels=2) #target still has 2 channels
        
        if mask is not None:
            masked_input=get_masked_input_img_from_tensor(input[idx],mask)


        # Display the image using matplotlib
         #print("image shape: ", image_np.shape)
        #define the nr of subplots 
        if target_images is not None and mask is not None:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        elif target_images is not None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))     
        else:
            fix, ax = plt.subplots(1, 1, figsize=(5, 5))  

        ax[0].imshow(output) 
        ax[0].set_title('Output')
        ax[0].axis('off')  # Hide axes

        if target_images is not None:
             # Plot the 'target' image
            ax[1].imshow(target) 
            ax[1].set_title('Target')
            ax[1].axis('off')  # Hide axes

        if mask is not None:
            ax[2].imshow(masked_input)
            ax[2].set_title('Masked Input')
            ax[2].axis('off')    

        plt.savefig(f'imgOut/{name}_img_{idx}.png')
        plt.close()
    