import functools
from enum import Enum
from typing import Any, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.models import (ResNet, ResNet18_Weights, ResNet50_Weights,
                                WeightsEnum, resnet18, resnet50)
from torchvision.models.resnet import BasicBlock, Bottleneck

###############################################################################
# Helper Functions
###############################################################################


def prediction(out, target=None, onehot=True):
  if target is None:
    _, label = torch.max(out.data, 1)
    return label
  else:
    if onehot:
      _, label = torch.max(out.data, 1)
    else:  # if output is a one channel, set a label where threshold is 0.5
      label = torch.where(out.data > torch.FloatTensor([0.5]), torch.ones(
          out.size()[0]).long(), torch.zeros(out.size()[0]).long())
    acc = (label == target).sum().item() / target.size()[0]
    return label, acc


def get_norm_layer(norm_type='instance'):
  """Return a normalization layer

  Parameters:
      norm_type (str) -- the name of the normalization layer: batch | instance | none

  For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
  For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
  """
  if norm_type == 'batch':
    norm_layer = functools.partial(
        nn.BatchNorm2d, affine=True, track_running_stats=True)
  elif norm_type == 'instance':
    norm_layer = functools.partial(
        nn.InstanceNorm2d, affine=False, track_running_stats=False)
  elif norm_type == 'cond_instance':
    norm_layer = CondInstanceNorm
  elif norm_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError(
        'normalization layer [%s] is not found' % norm_type)
  return norm_layer


def get_scheduler(optimizer, opt):
  """Return a learning rate scheduler

  Parameters:
      optimizer          -- the optimizer of the network
      opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                            opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

  For 'linear', we keep the same learning rate for the first <opt.niter> epochs
  and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
  For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
  See https://pytorch.org/docs/stable/optim.html for more details.
  """
  if opt.lr_policy == 'linear':
    def lambda_rule(epoch):
      lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                       opt.niter) / float(opt.niter_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif opt.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
  elif opt.lr_policy == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
  elif opt.lr_policy == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.niter, eta_min=0)
  else:
    return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
  return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
  """Initialize network weights.

  Parameters:
      net (network)   -- network to be initialized
      init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
      init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

  We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
  work better for some applications. Feel free to try yourself.
  """
  def init_func(m):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      if init_type == 'normal':
        init.normal_(m.weight.data, 0.0, init_gain)
      elif init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=init_gain)
      elif init_type == 'kaiming':
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif init_type == 'orthogonal':
        init.orthogonal_(m.weight.data, gain=init_gain)
      else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)
      if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, init_gain)
      init.constant_(m.bias.data, 0.0)

  print('initialize network with %s' % init_type)
  net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
  """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
  Parameters:
      net (network)      -- the network to be initialized
      init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
      gain (float)       -- scaling factor for normal, xavier and orthogonal.
      gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

  Return an initialized network.
  """
  if len(gpu_ids) > 1:  # and torch.cuda.device_count() > 1
    assert (torch.cuda.is_available())
    net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    net.to(gpu_ids[0])
    print("use multi gpus")
  elif len(gpu_ids) == 1:
    assert (torch.cuda.is_available())
    net.cuda()
    print("use single gpu")
  init_weights(net, init_type, init_gain=init_gain)
  return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
  """Create a generator

  Parameters:
      input_nc (int) -- the number of channels in input images
      output_nc (int) -- the number of channels in output images
      ngf (int) -- the number of filters in the last conv layer
      netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
      norm (str) -- the name of normalization layers used in the network: batch | instance | none
      use_dropout (bool) -- if use dropout layers.
      init_type (str)    -- the name of our initialization method.
      init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
      gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

  Returns a generator

  Our current implementation provides two types of generators:
      U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
      The original U-Net paper: https://arxiv.org/abs/1505.04597

      Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
      Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
      We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


  The generator has been initialized by <init_net>. It uses RELU for non-linearity.
  """
  net = None
  norm_layer = get_norm_layer(norm_type=norm)

  if netG == 'resnet_9blocks':
    net = ResnetGenerator(input_nc, output_nc, ngf,
                          norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
  elif netG == 'resnet_6blocks':
    net = ResnetGenerator(input_nc, output_nc, ngf,
                          norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
  elif netG == 'resnet_2blocks':
    net = ResnetGenerator(input_nc, output_nc, ngf,
                          norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=2)
  elif netG == 'unet_128':
    net = UnetGenerator(input_nc, output_nc, 7, ngf,
                        norm_layer=norm_layer, use_dropout=use_dropout)
  elif netG == 'unet_256':
    net = UnetGenerator(input_nc, output_nc, 8, ngf,
                        norm_layer=norm_layer, use_dropout=use_dropout)
  else:
    raise NotImplementedError(
        'Generator model name [%s] is not recognized' % netG)
  return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
  """Create a discriminator

  Parameters:
      input_nc (int)     -- the number of channels in input images
      ndf (int)          -- the number of filters in the first conv layer
      netD (str)         -- the architecture's name: basic | n_layers | pixel
      n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
      norm (str)         -- the type of normalization layers used in the network.
      init_type (str)    -- the name of the initialization method.
      init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
      gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

  Returns a discriminator

  Our current implementation provides three types of discriminators:
      [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
      It can classify whether 70×70 overlapping patches are real or fake.
      Such a patch-level discriminator architecture has fewer parameters
      than a full-image discriminator and can work on arbitrarily-sized images
      in a fully convolutional fashion.

      [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
      with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

      [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
      It encourages greater color diversity but has no effect on spatial statistics.

  The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
  """
  net = None
  norm_layer = get_norm_layer(norm_type=norm)

  if netD == 'basic':  # default PatchGAN classifier
    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
  elif netD == 'n_layers':  # more options
    net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
  elif netD == 'pixel':     # classify if each pixel is real or fake
    net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
  elif netD == "latent":
    net = LatentDiscriminator(input_nc, ndf, norm_layer=norm_layer)
  else:
    raise NotImplementedError(
        'Discriminator model name [%s] is not recognized' % net)
  return init_net(net, init_type, init_gain, gpu_ids)


def define_C(input_nc, netC, init_type='normal', init_gain=0.02, gpu_ids=[]):
  if netC == "cycada" or netC == "lenet":
    net = VanillaClassifier(num_classes=3, bottleneck_dim=256)
  elif netC == "lenet28":  # for 28
    net = LeNet28(input_nc)
  elif netC == "dtn":  # for 32x32
    net = DTNClassifier(input_nc)
  elif netC == "d_ft":
    net = FeatureDiscriminator()
  elif netC == "vanilla":
    net = VanillaClassifier(num_classes=3, bottleneck_dim=256)
  else:
    raise NotImplementedError
  return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################


class GANLoss(nn.Module):
  """Define different GAN objectives.

  The GANLoss class abstracts away the need to create the target label tensor
  that has the same size as the input.
  """

  def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
    """ Initialize the GANLoss class.

    Parameters:
        gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
        target_real_label (bool) - - label for a real image
        target_fake_label (bool) - - label of a fake image

    Note: Do not use sigmoid as the last layer of Discriminator.
    LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
    """
    super(GANLoss, self).__init__()
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))
    self.gan_mode = gan_mode
    if gan_mode == 'lsgan':
      self.loss = nn.MSELoss()
    elif gan_mode == 'vanilla':
      self.loss = nn.BCEWithLogitsLoss()
    elif gan_mode in ['wgangp']:
      self.loss = None
    elif gan_mode in ['wgan']:
      self.loss = None
    else:
      raise NotImplementedError('gan mode %s not implemented' % gan_mode)

  def get_target_tensor(self, prediction, target_is_real):
    """Create label tensors with the same size as the input.

    Parameters:
        prediction (tensor) - - tpyically the prediction from a discriminator
        target_is_real (bool) - - if the ground truth label is for real images or fake images

    Returns:
        A label tensor filled with ground truth label, and with the size of the input
    """

    if target_is_real:
      target_tensor = self.real_label
    else:
      target_tensor = self.fake_label
    return target_tensor.expand_as(prediction)

  def __call__(self, prediction, target_is_real):
    """Calculate loss given Discriminator's output and grount truth labels.

    Parameters:
        prediction (tensor) - - tpyically the prediction output from a discriminator
        target_is_real (bool) - - if the ground truth label is for real images or fake images

    Returns:
        the calculated loss.
    """
    if self.gan_mode in ['lsgan', 'vanilla']:
      target_tensor = self.get_target_tensor(prediction, target_is_real)
      loss = self.loss(prediction, target_tensor)
    elif self.gan_mode == 'wgangp':
      if target_is_real:
        loss = -prediction.mean()
      else:
        loss = prediction.mean()
    return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
  """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

  Arguments:
      netD (network)              -- discriminator network
      real_data (tensor array)    -- real images
      fake_data (tensor array)    -- generated images from the generator
      device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
      type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
      constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
      lambda_gp (float)           -- weight for this loss

  Returns the gradient penalty loss
  """
  if lambda_gp > 0.0:
    # either use real images, fake images, or a linear interpolation of two.
    if type == 'real':
      interpolatesv = real_data
    elif type == 'fake':
      interpolatesv = fake_data
    elif type == 'mixed':
      alpha = torch.rand(real_data.shape[0], 1)
      alpha = alpha.expand(real_data.shape[0], real_data.nelement(
      ) // real_data.shape[0]).contiguous().view(*real_data.shape)
      alpha = alpha.to(device)
      interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
      raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(
                                        disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) -
                        constant) ** 2).mean() * lambda_gp        # added eps
    return gradient_penalty, gradients
  else:
    return 0.0, None


class ResnetGenerator(nn.Module):
  """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

  We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
  """

  def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
    """Construct a Resnet-based generator

    Parameters:
        input_nc (int)      -- the number of channels in input images
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """
    assert (n_blocks >= 0)
    super(ResnetGenerator, self).__init__()
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
             norm_layer(ngf),
             nn.ReLU(True)]

    n_downsampling = 2
    for i in range(n_downsampling):  # add downsampling layers
      mult = 2 ** i
      model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)]

    mult = 2 ** n_downsampling
    for i in range(n_blocks):       # add ResNet blocks

      model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                            norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

    for i in range(n_downsampling):  # add upsampling layers
      mult = 2 ** (n_downsampling - i)
      model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                # nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),  # avoid artifact use ResConv instead of Transposed Conv
                # nn.ReflectionPad2d(1),                                                # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
                # nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
    model += [nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, input):
    """Standard forward"""
    return self.model(input)


class ResnetBlock(nn.Module):
  """Define a Resnet block"""

  def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
    """Initialize the Resnet block

    A resnet block is a conv block with skip connections
    We construct a conv block with build_conv_block function,
    and implement skip connections in <forward> function.
    Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(
        dim, padding_type, norm_layer, use_dropout, use_bias)

  def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
    """Construct a convolutional block.

    Parameters:
        dim (int)           -- the number of channels in the conv layer.
        padding_type (str)  -- the name of padding layer: reflect | replicate | zero
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
        use_bias (bool)     -- if the conv layer uses bias or not

    Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
    """
    conv_block = []
    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError(
          'padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                             bias=use_bias), norm_layer(dim), nn.ReLU(True)]
    if use_dropout:
      conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError(
          'padding [%s] is not implemented' % padding_type)
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                             padding=p, bias=use_bias), norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    """Forward function (with skip connections)"""
    out = x + self.conv_block(x)  # add skip connections
    return out


class UnetGenerator(nn.Module):
  """Create a Unet-based generator"""

  def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
    """Construct a Unet generator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                            image of size 128x128 will become of size 1x1 # at the bottleneck
        ngf (int)       -- the number of filters in the last conv layer
        norm_layer      -- normalization layer

    We construct the U-Net from the innermost layer to the outermost layer.
    It is a recursive process.
    """
    super(UnetGenerator, self).__init__()
    # construct unet structure
    unet_block = UnetSkipConnectionBlock(
        ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
    for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
      unet_block = UnetSkipConnectionBlock(
          ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
    # gradually reduce the number of filters from ngf * 8 to ngf
    unet_block = UnetSkipConnectionBlock(
        ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnectionBlock(
        ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnectionBlock(
        ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                         outermost=True, norm_layer=norm_layer)  # add the outermost layer

  def forward(self, input):
    """Standard forward"""
    return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
  """Defines the Unet submodule with skip connection.
      X -------------------identity----------------------
      |-- downsampling -- |submodule| -- upsampling --|
  """

  def __init__(self, outer_nc, inner_nc, input_nc=None,
               submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    """Construct a Unet submodule with skip connections.

    Parameters:
        outer_nc (int) -- the number of filters in the outer conv layer
        inner_nc (int) -- the number of filters in the inner conv layer
        input_nc (int) -- the number of channels in input images/features
        submodule (UnetSkipConnectionBlock) -- previously defined submodules
        outermost (bool)    -- if this module is the outermost module
        innermost (bool)    -- if this module is the innermost module
        norm_layer          -- normalization layer
        user_dropout (bool) -- if use dropout layers.
    """
    super(UnetSkipConnectionBlock, self).__init__()
    self.outermost = outermost
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d
    if input_nc is None:
      input_nc = outer_nc
    downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                         stride=2, padding=1, bias=use_bias)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(inner_nc)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(outer_nc)

    if outermost:
      upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1)
      down = [downconv]
      up = [uprelu, upconv, nn.Tanh()]
      model = down + [submodule] + up
    elif innermost:
      upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1, bias=use_bias)
      down = [downrelu, downconv]
      up = [uprelu, upconv, upnorm]
      model = down + up
    else:
      upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1, bias=use_bias)
      down = [downrelu, downconv, downnorm]
      up = [uprelu, upconv, upnorm]

      if use_dropout:
        model = down + [submodule] + up + [nn.Dropout(0.5)]
      else:
        model = down + [submodule] + up

    self.model = nn.Sequential(*model)

  def forward(self, x):
    if self.outermost:
      return self.model(x)
    else:   # add skip connections
      return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
  """Defines a PatchGAN discriminator"""

  def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
    """Construct a PatchGAN discriminator

    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """
    super(NLayerDiscriminator, self).__init__()
    # no need to use bias as BatchNorm2d has affine parameters
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func != nn.BatchNorm2d
    else:
      use_bias = norm_layer != nn.BatchNorm2d

    kw = 4
    padw = 1
    sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                          stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):  # gradually increase the number of filters
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      if nf_mult == 8:
        stride = 1
      else:
        stride = 2
      sequence += [
          nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                    stride=stride, padding=padw, bias=use_bias),
          norm_layer(ndf * nf_mult),
          nn.LeakyReLU(0.2, True)
      ]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                  stride=1, padding=padw, bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
    ]
    self.model_ft = nn.Sequential(*sequence)
    # output 1 channel prediction map
    sequence = [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                          stride=1, padding=padw)]
    self.model_cls = nn.Sequential(*sequence)
    self.model_head = nn.Sequential(
        # code in the implementation expects image level discrimininator
        nn.Linear(in_features=29*29, out_features=1),
        nn.Sigmoid()
    )

  def forward(self, input, with_ft=False):
    """Standard forward."""
    ft = self.model_ft(input)
    out_one_channel = self.model_cls(ft)
    out_logits = torch.flatten(out_one_channel, start_dim=1)
    out = self.model_head(out_logits)

    if with_ft:
      return out, ft
    else:
      return out


class PixelDiscriminator(nn.Module):
  """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

  def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
    """Construct a 1x1 PatchGAN discriminator

    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        norm_layer      -- normalization layer
    """
    super(PixelDiscriminator, self).__init__()
    # no need to use bias as BatchNorm2d has affine parameters
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func != nn.InstanceNorm2d
    else:
      use_bias = norm_layer != nn.InstanceNorm2d

    self.net = [
        nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=1,
                  stride=1, padding=0, bias=use_bias),
        norm_layer(ndf * 2),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

    self.net = nn.Sequential(*self.net)

  def forward(self, input):
    """Standard forward."""
    return self.net(input)


class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)


class LeNet(nn.Module):  # for 32
  def __init__(self, input_nc):
    super(LeNet, self).__init__()

    sequence = [
        nn.Conv2d(input_nc, 20, kernel_size=5, stride=1, padding=0),
        nn.MaxPool2d(2, 2),
        nn.ReLU(True),

        nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
        nn.Dropout2d(0.5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(True),

        Flatten(),

        nn.Linear(50*5*5, 500),
        nn.ReLU(True),
        nn.Dropout(0.5),

        nn.Linear(500, 10)
    ]
    self.net = nn.Sequential(*sequence)

  def forward(self, input, d_feat=False):
    """Standard forward."""
    self.out = self.net(input)
    return self.out


class LeNet28(nn.Module):  # for 28
  def __init__(self, input_nc):
    super(LeNet28, self).__init__()

    sequence = [
        nn.Conv2d(input_nc, 20, kernel_size=5, stride=1, padding=0),
        nn.MaxPool2d(2, 2),
        nn.ReLU(True),

        nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
        nn.Dropout2d(0.5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(True),

        Flatten(),

        nn.Linear(50*4*4, 500),
        nn.ReLU(True),
        nn.Dropout(0.5),

        nn.Linear(500, 10)
    ]
    self.net = nn.Sequential(*sequence)

  def forward(self, input, d_feat=False):
    """Standard forward."""
    self.out = self.net(input)
    return self.out


class DTNClassifier(nn.Module):
  def __init__(self, input_nc):
    super(DTNClassifier, self).__init__()
    self.conv_params = nn.Sequential(
        nn.Conv2d(input_nc, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.Dropout2d(0.3),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.Dropout2d(0.5),
        nn.ReLU()
    )

    self.fc_params = nn.Sequential(
        nn.Linear(256*4*4, 512),
        nn.BatchNorm1d(512),
    )

    self.classifier = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 10)
    )

  def forward(self, x, with_ft=False):
    x = self.conv_params(x)
    x = x.view(x.size(0), -1)
    x = self.fc_params(x)
    score = self.classifier(x)
    if with_ft:
      return score, x
    else:
      return score


class FeatureDiscriminator(nn.Module):
  def __init__(self):
    super(FeatureDiscriminator, self).__init__()
    self.discriminator = nn.Sequential(
        nn.Linear(3, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 2),
    )

  def forward(self, score):
    out = self.discriminator(score)
    return out


class ResNetBackbone(ResNet):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.out_features = self.fc.in_features
    del self.avgpool
    del self.fc

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x


def get_resnet_backbone(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNetBackbone:

  model = ResNetBackbone(block, layers, **kwargs)

  if weights is not None:
    model_dict = model.state_dict()
    pretrained_dict = weights.get_state_dict(progress=progress)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
    }
    model.load_state_dict(pretrained_dict)

  return model


def resnet50_backbone(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any):
  weights = ResNet50_Weights.verify(weights)
  return get_resnet_backbone(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


class ResNetType(Enum):
  ResNet18 = 18
  ResNet50 = 50


backbones = {
    ResNetType.ResNet50: resnet50_backbone
}
pretrained_backbone_weights = {
    ResNetType.ResNet50: ResNet50_Weights.IMAGENET1K_V2
}


class FeatureExtractor(nn.Module):
  def __init__(
      self,
      backbone_type: ResNetType = ResNetType.ResNet50
  ) -> None:
    super().__init__()

    self.backbone = backbones[backbone_type]()
    self.avgpool = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1)
    )

  def forward(self, x):
    x = self.backbone(x)
    x = self.avgpool(x)
    return x


def feature_extractor(
    backbone_type: ResNetType = ResNetType.ResNet50,
    backbone_pretrained: bool = True
):
  feature_extractor = FeatureExtractor(backbone_type=backbone_type)
  if backbone_pretrained:
    feature_extractor.backbone = backbones[backbone_type](
        weights=pretrained_backbone_weights[backbone_type]
    )
  return feature_extractor


class VanillaClassifier(nn.Module):
  def __init__(
      self,
      num_classes: int = 3,
      backbone_type: ResNetType = ResNetType.ResNet50,
      backbone_pretrained: bool = True,
      bottleneck_dim: Optional[int] = 256
  ) -> None:
    super().__init__()

    self.feature_extractor = feature_extractor(
        backbone_type=backbone_type,
        backbone_pretrained=backbone_pretrained
    )
    self.bottleneck = nn.Identity()
    if bottleneck_dim:
      self.bottleneck = nn.Sequential(
          nn.Linear(self.feature_extractor.backbone.out_features, bottleneck_dim),
          nn.BatchNorm1d(bottleneck_dim),
          nn.ReLU()
      )
    self.head = nn.Linear(
        in_features=(
            bottleneck_dim if bottleneck_dim
            else self.feature_extractor.backbone.out_features
        ),
        out_features=num_classes
    )

  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.bottleneck(x)
    x = self.head(x)
    return x
