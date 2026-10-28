"""Reusable PyTorch building blocks for the ECG VAE encoders/decoders.

    - basic blocks: ResizeConv2d, LinearResidualBlock, BasicBlockEnc/Dec,
      Reshape
    - residual blocks: 2D (ResidualBlock, ResidualBlock2D2, transpose
      variants) and 1D (ResidualBlock1D, ResidualBlock1D2, transpose
      variants)
    - feature layers: Res64 / Res64Transpose, Layer64 / Layer64Transpose
      (repeated down/up-sampling convs)
    - GradientReversal: gradient reversal layer (domain-adversarial
      training)
    - helpers: to_device, is_list_or_tuple, add_to_recordable_attributes
"""

import torch
from torch import nn
import torch.nn.functional as F


class ResizeConv2d(nn.Module):
    """Upscale with F.interpolate followed by a Conv2d (decoder)."""


    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

# basis linear residual block
class LinearResidualBlock(nn.Module):
    """Linear residual block: two linear layers + LeakyReLU, identity shortcut."""

    
    def __init__(self, features):
        super(LinearResidualBlock, self).__init__()
        self.linear1 = nn.Linear(features, features)
        # self.bn1 = nn.BatchNorm1d(features)
        self.relu = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(features, features)
        # self.bn2 = nn.BatchNorm1d(features)
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        # out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    


        
class BasicBlockEnc(nn.Module):
    """2D residual (ResNet-style) encoder block with learnable shortcut."""


    def __init__(self, in_planes, stride=(1, 2)):
        super().__init__()

        if in_planes == 512:
            planes = in_planes
        else:
            planes = in_planes*stride[1]

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    """2D residual decoder block; up-sampling done with ResizeConv2d."""


    def __init__(self, in_planes, stride=(1, 1), layer=1):
        super().__init__()

        if in_planes == 512 and layer == 2:
            planes = in_planes
        else:
            planes = int(in_planes/stride[1])

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class Reshape(nn.Module):
    """Module that reshapes a tensor to a fixed shape (usable in nn.Sequential)."""

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ResidualBlock(nn.Module):
    """2D residual block with a learnable 1x1 downsample path."""

    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = (1, 3), padding = (0, 1), downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = (1, 1), padding = padding),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels
        self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=(0,0)),
                nn.BatchNorm2d(out_channels),
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample or self.out_channels != residual.shape[1]:
            residual = self.downsample_layer(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualBlock2D2(nn.Module):
    """2D residual block (conv-bn-relu x2) with 1x1 conv for shape matching."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock2D2, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels differ, a 1x1 conv is needed to match the dimensions
        if in_channels != out_channels or stride != 1:
            self.residual_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x):
        # Save the input for the skip connection
        residual = self.residual_connection(x)

        # First conv layer with activation and normalization
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv layer with activation and normalization
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the residual (skip connection)
        out += residual
        out = self.relu(out)

        return out

class ResidualBlock2D_2_Transpose(nn.Module):
    """2D residual block using transposed convolutions (decoder path)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock2D_2_Transpose, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If the input and output channels differ, a 1x1 conv is needed to match the dimensions
        if in_channels != out_channels or stride != 1:
            self.residual_connection = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
        else:
            self.residual_connection = nn.Identity()
            
    def forward(self, x):
        # Save the input for the skip connection
        residual = self.residual_connection(x)
        
        # First conv layer with activation and normalization
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv layer with activation and normalization
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual (skip connection)
        out += residual
        out = self.relu(out)
        
        return out

class ResidualBlock1D2(nn.Module):
    """1D residual block (conv-bn-relu x2) with 1x1 conv for shape matching."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D2, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If the input and output channels differ, a 1x1 conv is needed to match the dimensions
        if in_channels != out_channels or stride != 1:
            self.residual_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x):
        # Save the input for the skip connection
        residual = self.residual_connection(x)

        # First conv layer with activation and normalization
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv layer with activation and normalization
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the residual (skip connection)
        out += residual
        out = self.relu(out)

        return out

class ResidualBlock1D_2_Transpose(nn.Module):
    """1D residual block using transposed convolutions (decoder path)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D_2_Transpose, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # If the input and output channels differ, a 1x1 conv is needed to match the dimensions
        if in_channels != out_channels or stride != 1:
            self.residual_connection = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
        else:
            self.residual_connection = nn.Identity()
            
    def forward(self, x):
        # Save the input for the skip connection
        residual = self.residual_connection(x)
        
        # First conv layer with activation and normalization
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv layer with activation and normalization
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual (skip connection)
        out += residual
        out = self.relu(out)
        
        return out

class ResidualBlock1D(nn.Module):
    """1D residual block with an optional 1x1 conv downsample layer."""

    def __init__(self, in_channels, out_channels, stride = 2, kernel_size = 3, padding = 1, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm1d(out_channels),
                        nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels
        self.downsample_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels),
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample or self.out_channels != residual.shape[1]:
            residual = self.downsample_layer(x)
        out += residual
        out = self.relu(out)
        return out

class Res64(nn.Module):
    """Two same-width conv layers applied twice + identity shortcut (encoder)."""

    def __init__(self, k, p, l=64, conv2d=False, conv2d_mix=False):#, k, p, n, , conv1d_12=False):
        super().__init__()
        self.k = k
        self.p = p
        # self.n = n
        # self.conv1d_12 = conv1d_12

        if conv2d:
            self.layer = nn.Sequential(
                nn.Conv2d(l, l, stride=(1, 1), kernel_size=(1, self.k), padding=(0, self.p)),
                nn.BatchNorm2d(l),
                nn.LeakyReLU(0.01))
        elif conv2d_mix:
            self.layer = nn.Sequential(
                nn.Conv2d(l, l, stride=(1, 1), kernel_size=(3, self.k), padding=(1, self.p)),
                nn.BatchNorm2d(l),
                nn.LeakyReLU(0.01))

        # elif self.conv1d_12:
        #     self.layer = nn.Sequential(
        #         nn.Conv1d(64*12, 64*12, stride=(2), kernel_size=self.k, padding=self.p, groups=12),
        #         nn.LeakyReLU(0.01))
        # else:
        else:
            self.layer = nn.Sequential(
                nn.Conv1d(l, l, stride=1, kernel_size=self.k, padding=self.p),
                nn.BatchNorm1d(l),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        identity = x

        out = self.layer(x)
        out = self.layer(out)
        out += identity

        return out


class ResidualBlockTranspose(nn.Module):
    """2D residual block built from transposed convolutions (decoder)."""

    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = (1, 3), padding = (0, 1), upsample = None):
        super(ResidualBlockTranspose, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                        nn.ConvTranspose2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
                        nn.BatchNorm2d(out_channels))
        self.upsample = upsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels
        self.upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=(0,0)),
                nn.BatchNorm2d(out_channels),
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.upsample or self.out_channels != residual.shape[1]:
            residual = self.upsample_layer(x)
        out += residual
        out = self.relu(out)
        return out

class Res64Transpose(nn.Module):
    """Transposed (up-sampling) counterpart of Res64, for the decoder."""

    def __init__(self, k, p, l=64, conv2d=False, conv2d_mix=False):#, k, p, n, conv2d=False, conv1d_12=False):
        super().__init__()
        self.k = k
        self.p = p
        # self.n = n

        if conv2d:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(l, l, stride=(1, 1), kernel_size=(1, self.k), padding=(0, self.p)),
                # nn.BatchNorm2d(l),
                nn.LeakyReLU(0.01))
        elif conv2d_mix:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(l, l, stride=(1, 1), kernel_size=(3, self.k), padding=(1, self.p)),
                # nn.BatchNorm2d(l),
                nn.LeakyReLU(0.01))
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose1d(l, l, stride=1, kernel_size=self.k, padding=self.p),
                # nn.BatchNorm1d(l),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        identity = x

        out = self.layer(x)
        out = self.layer(out)
        out += identity

        return out


class Layer64(nn.Module):
    """Repeat a strided (down-sampling) conv layer n times."""

    def __init__(self, k, p, n, conv2d=False, conv1d_12=False, c=64):
        super().__init__()
        self.k = k
        self.p = p
        self.n = n
        self.conv2d = conv2d
        self.conv1d_12 = conv1d_12

        if self.conv2d:
            self.layer = nn.Sequential(
                nn.Conv2d(c, c, stride=(1, 2), kernel_size=(1, self.k), padding=(0, self.p)),
                # nn.BatchNorm2d(64, eps=1e-02),
                nn.LeakyReLU(0.01))
        elif self.conv1d_12:
            self.layer = nn.Sequential(
                nn.Conv1d(c*12, c*12, stride=(2), kernel_size=self.k, padding=self.p, groups=12),
                nn.LeakyReLU(0.01))
        else:
            self.layer = nn.Sequential(
                nn.Conv1d(c, c, stride=(2), kernel_size=self.k, padding=self.p),
                # nn.BatchNorm1d(64, eps=1e-02),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        for _ in range(self.n):
            x = self.layer(x)
        return x


class Layer64Transpose(nn.Module):
    """Repeat a transposed (up-sampling) conv layer n times."""

    def __init__(self, k, p, n, conv2d=False, conv1d_12=False, c=64):
        super().__init__()
        self.k = k
        self.p = p
        self.n = n
        self.conv2d = conv2d
        self.conv1d_12 = conv1d_12

        if self.conv2d:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(c, c, stride=(1, 2), kernel_size=(1, self.k), padding=(0, self.p)),
                # nn.BatchNorm2d(c, eps=1e-02),
                nn.LeakyReLU(0.01))

        elif self.conv1d_12:
            self.layer = nn.Sequential(
                nn.ConvTranspose1d(c*12, c*12, stride=(2), kernel_size=self.k, padding=self.p, groups=12),
                nn.LeakyReLU(0.01))
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose1d(c, c, stride=(2), kernel_size=self.k, padding=self.p),
                # nn.BatchNorm1d(c, eps=1e-02),
                nn.LeakyReLU(0.01))

    def forward(self, x):
        for _ in range(self.n):
            x = self.layer(x)
        return x

def extra_repr(cls, attr_names, delimiter="\n"):
    return delimiter.join([f"{a}={str(getattr(cls, a))}" for a in attr_names])


def is_list_or_tuple(x):
# Check whether x is a list or tuple

    return isinstance(x, (list, tuple))


def add_to_recordable_attributes(
    input_obj, name=None, list_of_names=None, is_stat=False
):
    """Register attribute names that should be reported by logging utilities."""

    if is_stat:
        attr_name_list_name = "_record_these_stats"
    else:
        attr_name_list_name = "_record_these"
    if not hasattr(input_obj, attr_name_list_name):
        setattr(input_obj, attr_name_list_name, [])
    attr_name_list = getattr(input_obj, attr_name_list_name)
    if name is not None:
        if name not in attr_name_list:
            attr_name_list.append(name)
        if not hasattr(input_obj, name):
            setattr(input_obj, name, 0)
    if list_of_names is not None and is_list_or_tuple(list_of_names):
        for n in list_of_names:
            add_to_recordable_attributes(input_obj, name=n, is_stat=is_stat)


class _GradientReversal(torch.autograd.Function):
    """autograd: identity forward, negates/scales the gradient in backward."""

    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def to_dtype(x, dtype=None):
    """Cast tensor x to the given dtype (helper used by to_device)."""
    return x.to(dtype)


def to_device(x, tensor=None, device=None, dtype=None):
    """Move tensor x onto the given device/dtype when needed (else no-op)."""

    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x

class GradientReversal(nn.Module):
    """
    Implementation of the gradient reversal layer described in
    [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818),
    which 'leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it
    by a negative scalar during backpropagation.'
    """

    def __init__(self, weight: float = 1.0):
        """
        Arguments:
            weight: The gradients  will be multiplied by ```-weight```
                during the backward pass.
        """
        super().__init__()
        self.register_buffer("weight", torch.tensor([weight]))
        add_to_recordable_attributes(self, "weight")

    def update_weight(self, new_weight):
        self.weight[0] = new_weight

    def forward(self, x):
        """"""
        return _GradientReversal.apply(x, to_device(self.weight, x))

    def extra_repr(self):
        """"""
        return extra_repr(self, ["weight"])