import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def create_swir_img_from_tensor(image): 
    image = image.cpu().numpy()
    npimgtransposed = np.transpose(image, (1, 2, 0))
    stacked_image = np.zeros((npimgtransposed.shape[0], npimgtransposed.shape[1], 3))
    
    if npimgtransposed.shape[2] == 2:
        # Assign the red channel to the first channel
        np.copyto(stacked_image[:, :, 0], npimgtransposed[:,:,0])
        # Assign the green channel to the second channel
        np.copyto(stacked_image[:, :, 1], npimgtransposed[:,:,1])   
    elif npimgtransposed.shape[2] == 3:
           # Assign the red channel to the first G0 channel
        np.copyto(stacked_image[:, :, 0], npimgtransposed[:,:,0])
        # Assign the green channel to the second G0 channel
        np.copyto(stacked_image[:, :, 1], npimgtransposed[:,:,1])
        # Assign the blue channel to the third G0 channel
        np.copyto(stacked_image[:, :, 2], npimgtransposed[:,:,2])
        
    return stacked_image

def get_binary_mask_from_tensor(mask):  # mask shape: [3,144] - [Channel Groups, 12*12 patches] - [:,i] --> has 0 (keep) or 1 (masked)
    mask =mask.cpu().numpy()
    num_patches_per_axis = int(mask.shape[1] ** 0.5)
    print("patches per axis: ", num_patches_per_axis)   
    bin_mask = mask.reshape(num_patches_per_axis,num_patches_per_axis)    
    return bin_mask

def get_masked_input_img_from_tensor(input,mask,group=2):  # input shape: [3,96,96] - mask shape: [3,144] - [Channel Groups, 12*12 patches] - [:,i] --> has 0 (keep) or 1 (masked)

    swir_input_full=create_swir_img_from_tensor(input) #shape [96,96,3]
    input_size = swir_input_full.shape[0]   
    if group == 2:
        binary_mask = get_binary_mask_from_tensor(mask,group=2)               #shape [12,12]
    elif group == 1: 
        binary_mask = get_binary_mask_from_tensor(mask,group=1)  
    elif group == 0:                                                             #shape [12,12]
        binary_mask = get_binary_mask_from_tensor(mask,group=0)  

    # resize the binary mask to [96, 96] (nearest-neighbor interpolation)
    resized_mask = resize(binary_mask, (input_size, input_size), order=0, preserve_range=True)

    # Apply the resized mask to the image
    # Remember, zeros are 'keep', so no need to invert the mask
    masked_image = swir_input_full * (1 - resized_mask[..., np.newaxis])  # invert and Apply the mask across all channels

    return masked_image


def save_input_output_fig(final_swir_images ,name , target_images , mask=None,input=None ):  # final_image shape: [8,2,96,96]
    

    for idx, img in enumerate(final_swir_images) :
    
        swir_output = create_swir_img_from_tensor(img)

        input_g0 = get_masked_input_img_from_tensor(input[idx,[0,1,2]],mask=mask[0])
        input_g1 = get_masked_input_img_from_tensor(input[idx,[3,4,5]],mask=mask[1])
        input_g2 = get_masked_input_img_from_tensor(input[idx,[8,9]],mask=mask[2])

    
        swir_target = create_swir_img_from_tensor(target_images[idx]) #target still has 2 channels
        
        # if mask is not None and input is not None:
        #     masked_input=get_masked_input_img_from_tensor(input[idx],mask[idx])
        #     # masked_input=mask[idx,2,:].cpu().numpy()
        #     # masked_input = masked_input.reshape(12,12)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0].set_title('Input')
    ax[0, 0].imshow(input_g2)
    ax[0, 0].set_title('SWIR')
    ax[0,1].imshow(input_g1, 'Group 1')
    ax[0,2].imshow(input_g0, 'Group 0')
    ax[1].set_title('SWIR only')
    ax[1, 0].imshow(swir_output)
    ax[1, 0].set_title('Output')
    ax[1, 1].imshow(swir_target)
    ax[1, 1].set_title('Target')
  
    plt.savefig(f'imgOut/{name}_img_{idx}.png')
    plt.close()



def save_comparison_fig_from_tensor(final_swir_images ,name , target_images=None, mask=None,input=None ):  # final_image shape: [8,2,96,96]


    
    for idx, img in enumerate(final_swir_images) :
    
        output = create_swir_img_from_tensor(img)
        if target_images is not None:
            target = create_swir_img_from_tensor(target_images[idx]) #target still has 2 channels
        
        if mask is not None:
            masked_input=get_masked_input_img_from_tensor(input[idx],mask[idx])
            # masked_input=mask[idx,2,:].cpu().numpy()
            # masked_input = masked_input.reshape(12,12)


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

        fig.suptitle("SWIR Channel Comparison", fontsize=16)     

        plt.savefig(f'imgOut/{name}_img_{idx}.png')
        plt.close()
    