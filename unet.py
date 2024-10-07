import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class deform_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1,off_set_groups=2):
        super(deform_conv, self).__init__()
        off_ch = 2*kernel_size*kernel_size*off_set_groups
        mask_ch = off_set_groups*kernel_size*kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size = kernel_size,stride=1,padding=padding)       
        self.conv_offset =nn.Conv2d(in_ch,off_ch,kernel_size = kernel_size,stride=1,padding=padding)
        init_offset = torch.Tensor(off_ch,in_ch,kernel_size,kernel_size)
        self.conv_offset.weight = torch.nn.Parameter(init_offset)
        self.conv_mask = nn.Conv2d(in_ch,mask_ch, kernel_size=kernel_size, stride = 1, padding=padding)
        init_mask = torch.zeros([mask_ch,in_ch,kernel_size,kernel_size])+0.5
        self.conv_mask.weight = torch.nn.Parameter(init_mask)
    def forward(self, x):        
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = deform_conv2d(input =x, offset=offset, weight = self.conv.weight,mask=mask,padding=(self.padding,self.padding))
        if torch.isnan(out.mean()) and not torch.isnan(x.mean()):
            print("Deformable Conv generated NaN")
            
        return out


class resnet_block(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch,int_ch=128):
        super(resnet_block, self).__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, int_ch, 1, padding=0),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, int_ch, 3, padding=1),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inc_block(nn.Module):    

    def __init__(self, channels, out_ch):
        super(inc_block, self).__init__()
        int_ch = int(channels/2)       
        self.conv = nn.Sequential(
            nn.Conv2d(channels, int_ch, 1, padding=0),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, int_ch, 3, padding=1),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, int_ch, 5, padding=2),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, int_ch, 3, padding=1),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),                        
            nn.Conv2d(int_ch, channels, 1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.adjust =  nn.Sequential(
            nn.Conv2d(channels, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),    
        )

    def forward(self, x):
        x = self.conv(x)+x        
        return self.adjust(x)

class resnet_block3(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, channels, out_ch):
        super(resnet_block2, self).__init__()
        int_ch = int(channels/2)       
        self.conv = nn.Sequential(
            nn.Conv2d(channels, int_ch, 1, padding=0),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, int_ch, 3, padding=1),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(int_ch, int_ch, 3, padding=1),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),            
            nn.Conv2d(int_ch, channels, 1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.adjust =  nn.Sequential(
            nn.Conv2d(channels, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),    
        )

    def forward(self, x):
        x = self.conv(x)+x        
        return self.adjust(x)

class deform_block(nn.Module):
     def __init__(self, in_ch):
        out_ch = in_ch//2
        super(deform_block,self).__init__()
        self.short_range = nn.Sequential(
            deform_conv(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.wide_range = nn.Sequential(
            deform_conv(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

     def forward(self,x):
        sr = self.short_range(x)
        wr = self.wide_range(x)
        return torch.cat((sr,wr),dim=1)

class resnet_block2(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, channels, out_ch):
        super(resnet_block2, self).__init__()
        int_ch = int(channels/2)       
        self.conv = nn.Sequential(
            nn.Conv2d(channels, int_ch, 1, padding=0),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            #deform_block(int_ch),
            DeformableConv2d(int_ch,int_ch,3,padding=1),
            #deform_conv(int_ch, int_ch, 3, padding=1),
            nn.BatchNorm2d(int_ch),
            nn.ReLU(inplace=True),
            #deform_conv(int_ch, int_ch, 3, padding=1),
            #nn.BatchNorm2d(int_ch),
            #nn.ELU(inplace=True),            
            nn.Conv2d(int_ch, channels, 1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.adjust =  nn.Sequential(
            nn.Conv2d(channels, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),    
        )

    def forward(self, x):
        x = self.conv(x)+x        
        return self.adjust(x)

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down2, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), resnet_block2(in_ch,out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up2_noSkip(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up2_noSkip, self).__init__()
        if bilinear:
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up2 = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = resnet_block2(in_ch,out_ch)
    def forward(self, x,shape):
        x = self.up2(x)

        # input is CHW
        diffY = shape[2] - x.size()[2]
        diffX = shape[3] - x.size()[3]

        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        return self.conv(x)

class up2(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up2, self).__init__()

        if bilinear:
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up2 = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = resnet_block2(in_ch,out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up2(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class down3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down3, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), resnet_block(in_ch,out_ch),)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up3(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up3, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = resnet_block(in_ch,out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class flatCNN(nn.Module):
    def __init__(self, n_channels, n_classes):      
        super(flatCNN, self).__init__()
        int_ch = 256
        self.mpconv = nn.Sequential( inc_block(n_channels,int_ch),inc_block(int_ch,int_ch),inc_block(int_ch,int_ch))
        self.outc = outconv(int_ch, n_classes)
    def forward(self, x):
        x = self.mpconv(x)
        return self.outc(x)
    


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        in_size = 128
        self.inc = inconv(n_channels, in_size)
        self.down1 = down(in_size, in_size)
        self.down2 = down(in_size, 2*in_size)
        self.down3 = down(2*in_size, 4*in_size)
        self.down4 = down(4*in_size,4*in_size)
        self.up1 = up(8*in_size, 2*in_size, False)#512+512
        self.up2 = up(4*in_size, in_size, False)#
        self.up3 = up(2*in_size, in_size, False)#128+128
        self.up4 = up(2*in_size, in_size, False) #64+64
        self.outc = outconv(in_size, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
class ShallowUNet(nn.Module):
    def __init__(self, n_channels, n_classes,channels=256):
        super(ShallowUNet, self).__init__()
        in_size = channels
        self.inc = double_conv(n_channels, in_size,kernel_size=1, padding=0)
        self.down1 = down2(in_size, in_size*2)
        self.down2 = down2(in_size*2, in_size*4)
        self.down3 = down2(in_size*4, in_size*4)
        #self.middle = resnet_block2(in_size*4,in_size*4)
        self.up1 = up2(in_size*8, in_size*2, False) 
        self.up2 = up2(in_size*4, in_size, False) 
        self.up3 = up2(in_size*2, in_size, False) 
        self.outc = outconv(in_size, n_classes)#
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.25)
        #self.dropout3 = nn.Dropout(0.25)
        #self.dropout4 = nn.Dropout(0.25)



    def forward(self, x):
        x1 = self.inc(x)
        #x12 = self.dropout1(x1)
        x2 = self.down1(x1)
        #x22 = self.dropout1(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x4= self.middle(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        #x = self.dropout3(x)
        x = self.up3(x, x1)
        #x = self.dropout4(x)
        x = self.outc(x)
        return x
    
    
class ShallowUNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ShallowUNet2, self).__init__()
        in_size = 256
        self.inc = double_conv(n_channels, in_size,kernel_size=1, padding=0)
        self.down1 = down3(in_size, in_size*2)
        self.down2 = down3(in_size*2, in_size*4)
        self.down3 = down3(in_size*4, in_size*4)
        #self.middle = resnet_block(in_size*4,in_size*4)
        self.up1 = up3(in_size*8, in_size*2, False) 
        self.up2 = up3(in_size*4, in_size, False) 
        self.up3 = up3(in_size*2, in_size, False) 
        self.outc = outconv(in_size, n_classes)#
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.25)
        #self.dropout3 = nn.Dropout(0.25)
        #self.dropout4 = nn.Dropout(0.25)



    def forward(self, x):
        x1 = self.inc(x)
        #x12 = self.dropout1(x1)
        x2 = self.down1(x1)
        #x22 = self.dropout1(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x4= x4+self.middle(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        #x = self.dropout3(x)
        x = self.up3(x, x1)
        #x = self.dropout4(x)
        x = self.outc(x)
        return x
