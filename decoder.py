import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple U-Net-like decoder
class Decoder(nn.Module):
    def __init__(self, embed_dim, output_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(8, output_channels, kernel_size=1)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)

        self.apply(initialize_weights)


    def forward(self, x):
        print("decoder input", x.shape) #([8, 1024, 12, 12])   
        # Upsample progressively
        x = F.relu(self.conv1(x))  # 12x12 -> 12x12     --> ([8, 512, 12, 12])
        print( "after relu 1", x.shape)
        x = self.upconv1(x)        # 12x12 -> 24x24     -->  ([8, 256, 24, 24])
        print( "after upconv1", x.shape)
        x = F.relu(self.conv2(x))  # 24x24 -> 24x24     -->  ([8, 128, 24, 24]) 
        print( "after relu 2", x.shape)
        x = self.upconv2(x)        # 24x24 -> 48x48     -->  ([8, 64, 48, 48])
        print( "after upconv2", x.shape)
        x = F.relu(self.conv3(x))  # 48x48 -> 48x48     -->  ([8, 32, 48, 48]) 
        print( "after relu 3", x.shape)
        x = self.upconv3(x)        # 48x48 -> 96x96     -->  ([8, 16, 96, 96])
        print( "after upconv3", x.shape)
        x = F.relu(self.conv4(x))  # 96x96 -> 96x96     -->  ([8, 8, 96, 96])
        print( "after relu 4", x.shape)
    
        x = self.conv5(x)          # 96x96 -> 96x96   -->  ([8, 2, 192, 192])
        print( "after conv5", x.shape)
        return x

# Assuming `vit_output` is the output from the ViT with shape [8, 144, 1024]

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

# Step 1: Decode ViT output
def reshape_vit_output(vit_output):
    batch_size, seq_length, embed_dim = vit_output.shape
    # Ensure seq_length is a perfect square
    grid_size = int(seq_length ** 0.5)
    assert grid_size * grid_size == seq_length, "Sequence length should be a perfect square"

    # Reshape to [batch_size, grid_size, grid_size, embed_dim]
    vit_output = vit_output.view(batch_size, grid_size, grid_size, embed_dim)

    # Permute to [batch_size, embed_dim, grid_size, grid_size] for convolution
    vit_output = vit_output.permute(0, 3, 1, 2)
    
    return vit_output

# Example usage from ViT
#vit_output = torch.randn(8, 144, 1024)  # [batch_size, seq_length, embed_dim]
#decoded_output = decode_vit_output(vit_output)

#print("Decoded output shape:", decoded_output.shape)  # Should be [8, 1024, 12, 12]

# Instantiate the decoder model
#decoder_model = Decoder(embed_dim=1024, output_channels=2)

# Pass the decoded output from ViT through the decoder
#final_output = decoder_model(decoded_output)
#print("Final output shape:", final_output.shape)  # Should be [8, 2, 96, 96]
