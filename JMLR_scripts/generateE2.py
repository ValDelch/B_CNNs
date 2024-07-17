import torch
from e2cnn import gspaces
from e2cnn import nn

from BesselConv.BesselConv2d import BesselConv2d
from BesselConv.AttentiveNorm2d import AttentiveNorm2d
from BesselConv.GaussianBlur2d import GaussianBlur2d

def loadModel(dataset_name, model_name):
    if 'MNIST' in dataset_name:
        if model_name == 'E2_C4':
            
            """ Generate the model """

            # 115.067 parameters

            scale = 0.725
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=4)

                    in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 28, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=4, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x

        elif model_name == 'E2_C8':

            """ Generate the model """

            # 115.690 parameters

            scale = 0.525
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=8)

                    in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 28, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=4, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'E2_C16':

            """ Generate the model """

            # 102.836 parameters

            scale = 0.370
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=16)

                    in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 28, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=4, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
        
        elif model_name == 'small_SO2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=1, C_out=int(8*scale), k=9, padding='same', bias=True,
                                     reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5),
                        #Attention(int(8*scale)),
                        #torch.nn.BatchNorm2d(int(8*scale), eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(8*scale)),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True,
                                     reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5),
                        #Attention(int(16*scale)),
                        #torch.nn.BatchNorm2d(int(16*scale), eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(16*scale)),
                        torch.nn.Tanh(),
                        #torch.nn.Dropout(0.2),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                     reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5),
                        #Attention(int(24*scale)),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                     reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5),
                        #Attention(int(24*scale)),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.Tanh(),
                        #torch.nn.Dropout(0.2),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='same', bias=True,
                                     reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5),
                        #Attention(int(32*scale)),
                        #torch.nn.BatchNorm2d(int(32*scale), eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(32*scale)),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                     reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5),
                        #Attention(int(40*scale)),
                        #torch.nn.BatchNorm2d(int(40*scale), eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(40*scale)),
                        torch.nn.Tanh(),
                        #torch.nn.Dropout(0.5),
                    )

                    self.dense = torch.nn.Sequential(
                        #torch.nn.BatchNorm1d(int(40*scale), eps=1e-5, momentum=0.99),
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True),
                        #torch.nn.Softmax(dim=-1)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_cut':

            scale = 1.7

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=1, C_out=int(8*scale), k=9, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(8*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(16*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(32*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(40*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        #torch.nn.BatchNorm1d(int(40*scale), eps=1e-5, momentum=0.99),
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=1, C_out=int(8*scale), k=9, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(8*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(16*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(32*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(40*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        #torch.nn.BatchNorm1d(int(40*scale), eps=1e-5, momentum=0.99),
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=1, C_out=int(8*scale), k=9, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(8*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(16*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(32*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(40*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        #torch.nn.BatchNorm1d(int(40*scale), eps=1e-5, momentum=0.99),
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=1, C_out=int(8*scale), k=9, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(8*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(16*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(24*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(32*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        #torch.nn.BatchNorm2d(int(40*scale), eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        #torch.nn.BatchNorm1d(int(40*scale), eps=1e-5, momentum=0.99),
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_vanilla':

            scale = 0.9

            class CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(CNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=1, out_channels=int(8*scale), kernel_size=9, padding=4, bias=True),
                        torch.nn.BatchNorm2d(int(8*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(8*scale), out_channels=int(16*scale), kernel_size=7, padding=3, bias=True),
                        torch.nn.BatchNorm2d(int(16*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(16*scale), out_channels=int(24*scale), kernel_size=7, padding=3, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(24*scale), kernel_size=7, padding=3, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(32*scale), kernel_size=7, padding=3, bias=True),
                        torch.nn.BatchNorm2d(int(32*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(32*scale), out_channels=int(40*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(40*scale)),
                        torch.nn.ReLU(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'large_vanilla':

            # Extracted from https://github.com/rasbt/deeplearning-models/

            import torch.nn.functional as F

            def conv3x3(in_planes, out_planes, stride=1):
                """3x3 convolution with padding"""
                return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)

            class BasicBlock(torch.nn.Module):
                expansion = 1

                def __init__(self, inplanes, planes, stride=1, downsample=None):
                    super(BasicBlock, self).__init__()
                    self.conv1 = conv3x3(inplanes, planes, stride)
                    self.bn1 = torch.nn.BatchNorm2d(planes)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.conv2 = conv3x3(planes, planes)
                    self.bn2 = torch.nn.BatchNorm2d(planes)
                    self.downsample = downsample
                    self.stride = stride

                def forward(self, x):
                    residual = x

                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)

                    out = self.conv2(out)
                    out = self.bn2(out)

                    if self.downsample is not None:
                        residual = self.downsample(x)

                    out += residual
                    out = self.relu(out)

                    return out

            class CNN(torch.nn.Module):

                def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10, grayscale=True):
                    self.inplanes = 64
                    if grayscale:
                        in_dim = 1
                    else:
                        in_dim = 3
                    super(CNN, self).__init__()
                    self.conv1 = torch.nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
                    self.bn1 = torch.nn.BatchNorm2d(64)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    self.layer1 = self._make_layer(block, 64, layers[0])
                    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                    self.avgpool = torch.nn.AvgPool2d(7, stride=1)
                    self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

                    for m in self.modules():
                        if isinstance(m, torch.nn.Conv2d):
                            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                            m.weight.data.normal_(0, (2. / n)**.5)
                        elif isinstance(m, torch.nn.BatchNorm2d):
                            m.weight.data.fill_(1)
                            m.bias.data.zero_()

                def _make_layer(self, block, planes, blocks, stride=1):
                    downsample = None
                    if stride != 1 or self.inplanes != planes * block.expansion:
                        downsample = torch.nn.Sequential(
                            torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                            torch.nn.BatchNorm2d(planes * block.expansion),
                        )

                    layers = []
                    layers.append(block(self.inplanes, planes, stride, downsample))
                    self.inplanes = planes * block.expansion
                    for i in range(1, blocks):
                        layers.append(block(self.inplanes, planes))

                    return torch.nn.Sequential(*layers)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)

                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    # because MNIST is already 1x1 here:
                    # disable avg pooling
                    #x = self.avgpool(x)
                    
                    x = x.view(x.size(0), -1)
                    logits = self.fc(x)
                    #probas = F.softmax(logits, dim=1)
                    return logits

        elif model_name == 'E2_HN_1' or model_name == 'E2_HN_3' or model_name == 'E2_irr_1'  or model_name == 'E2_irr_3' or model_name == 'E2_irr_1_flip' or model_name == 'E2_irr_3_flip':

            """ Generate the model """

            # 115.690 parameters

            from ExpE2 import ExpE2SFCNN

        else:
            raise Exception("Model not implemented")

        """ Training params """

        if model_name == 'E2_HN_1':
            model = ExpE2SFCNN(n_channels=1, n_classes=10, N=-1, layer_type="hnet_conv", flip=False, scale=1.33)
        elif model_name == 'E2_HN_3':
            model = ExpE2SFCNN(n_channels=1, n_classes=10, N=-3, layer_type="hnet_conv", flip=False, scale=0.66)
        elif model_name == 'E2_irr_1':
            model = ExpE2SFCNN(n_channels=1, n_classes=10, N=-1, layer_type="realhnet", flip=False, scale=1.45)
        elif model_name == 'E2_irr_3':
            model = ExpE2SFCNN(n_channels=1, n_classes=10, N=-3, layer_type="realhnet", flip=False, scale=0.85)
        elif model_name == 'E2_irr_1_flip':
            model = ExpE2SFCNN(n_channels=1, n_classes=10, N=-1, layer_type="realhnet", flip=True, scale=1.1)
        elif model_name == 'E2_irr_3_flip':
            model = ExpE2SFCNN(n_channels=1, n_classes=10, N=-3, layer_type="realhnet", flip=True, scale=0.65)
        elif 'O2' in model_name:
            model = BCNN()
        elif 'vanilla' in model_name:
            model = CNN()
        else:
            model = E2CNN()

        # Epochs
        epochs = 50

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Loss function
        #loss_fcn = torch.nn.NLLLoss()
        loss_fcn = torch.nn.CrossEntropyLoss()

        # Metrics
        def torch_acc(y_pred, y_true):
            train_acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
            return train_acc

        metrics = [torch_acc]

        training_params = {'epochs': epochs, 'optimizer': optimizer, 'loss': loss_fcn, 'metrics': metrics}

        return model, training_params

    elif dataset_name == 'Galaxy' or dataset_name == 'Galaxy_noaug':
        if model_name == 'E2_C4':
            
            """ Generate the model """

            # 115.067 parameters

            scale = 0.725
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=4)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 128, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x

        elif model_name == 'E2_C8':

            """ Generate the model """

            # 115.690 parameters

            scale = 0.525
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=8)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 128, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'E2_C16':

            """ Generate the model """

            # 102.836 parameters

            scale = 0.370
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=16)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 128, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'small_SO2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_cut':

            scale = 1.7

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_vanilla':

            scale = 0.9

            class CNN(torch.nn.Module):
    
                def __init__(self, n_classes=10):
                    
                    super(CNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=3, out_channels=int(8*scale), kernel_size=9, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(8*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(8*scale), out_channels=int(16*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(16*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(16*scale), out_channels=int(24*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(24*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(32*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(32*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(32*scale), out_channels=int(40*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(40*scale)),
                        torch.nn.ReLU(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'large_vanilla':

            # Extracted from https://github.com/rasbt/deeplearning-models/

            import torch.nn.functional as F

            def conv3x3(in_planes, out_planes, stride=1):
                """3x3 convolution with padding"""
                return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)

            class BasicBlock(torch.nn.Module):
                expansion = 1

                def __init__(self, inplanes, planes, stride=1, downsample=None):
                    super(BasicBlock, self).__init__()
                    self.conv1 = conv3x3(inplanes, planes, stride)
                    self.bn1 = torch.nn.BatchNorm2d(planes)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.conv2 = conv3x3(planes, planes)
                    self.bn2 = torch.nn.BatchNorm2d(planes)
                    self.downsample = downsample
                    self.stride = stride

                def forward(self, x):
                    residual = x

                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)

                    out = self.conv2(out)
                    out = self.bn2(out)

                    if self.downsample is not None:
                        residual = self.downsample(x)

                    out += residual
                    out = self.relu(out)

                    return out

            class CNN(torch.nn.Module):

                def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10, grayscale=False):
                    self.inplanes = 64
                    if grayscale:
                        in_dim = 1
                    else:
                        in_dim = 3
                    super(CNN, self).__init__()
                    self.conv1 = torch.nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
                    self.bn1 = torch.nn.BatchNorm2d(64)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    self.layer1 = self._make_layer(block, 64, layers[0])
                    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                    #self.avgpool = torch.nn.AvgPool2d(7, stride=1)
                    self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

                    for m in self.modules():
                        if isinstance(m, torch.nn.Conv2d):
                            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                            m.weight.data.normal_(0, (2. / n)**.5)
                        elif isinstance(m, torch.nn.BatchNorm2d):
                            m.weight.data.fill_(1)
                            m.bias.data.zero_()

                def _make_layer(self, block, planes, blocks, stride=1):
                    downsample = None
                    if stride != 1 or self.inplanes != planes * block.expansion:
                        downsample = torch.nn.Sequential(
                            torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                            torch.nn.BatchNorm2d(planes * block.expansion),
                        )

                    layers = []
                    layers.append(block(self.inplanes, planes, stride, downsample))
                    self.inplanes = planes * block.expansion
                    for i in range(1, blocks):
                        layers.append(block(self.inplanes, planes))

                    return torch.nn.Sequential(*layers)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)

                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    # because MNIST is already 1x1 here:
                    # disable avg pooling
                    x = torch.mean(x, axis=(2,3))
                    
                    x = x.view(x.size(0), -1)
                    logits = self.fc(x)
                    #probas = F.softmax(logits, dim=1)
                    return logits

        elif model_name == 'E2_HN_1' or model_name == 'E2_HN_3' or model_name == 'E2_irr_1'  or model_name == 'E2_irr_3' or model_name == 'E2_irr_1_flip' or model_name == 'E2_irr_3_flip':

            """ Generate the model """

            # 115.690 parameters

            from ExpE2 import ExpE2SFCNN

        else:
            raise Exception("Model not implemented")

        """ Training params """

        if model_name == 'E2_HN_1':
            model = ExpE2SFCNN(n_channels=3, n_classes=10, N=-1, layer_type="hnet_conv", flip=False, scale=1.33)
        elif model_name == 'E2_HN_3':
            model = ExpE2SFCNN(n_channels=3, n_classes=10, N=-3, layer_type="hnet_conv", flip=False, scale=0.66)
        elif model_name == 'E2_irr_1':
            model = ExpE2SFCNN(n_channels=3, n_classes=10, N=-1, layer_type="realhnet", flip=False, scale=1.45)
        elif model_name == 'E2_irr_3':
            model = ExpE2SFCNN(n_channels=3, n_classes=10, N=-3, layer_type="realhnet", flip=False, scale=0.85)
        elif model_name == 'E2_irr_1_flip':
            model = ExpE2SFCNN(n_channels=3, n_classes=10, N=-1, layer_type="realhnet", flip=True, scale=1.1)
        elif model_name == 'E2_irr_3_flip':
            model = ExpE2SFCNN(n_channels=3, n_classes=10, N=-3, layer_type="realhnet", flip=True, scale=0.65)
        elif 'O2' in model_name:
            model = BCNN()
        elif 'vanilla' in model_name:
            model = CNN()
        else:
            model = E2CNN()

        # Epochs
        epochs = 50

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Loss function
        loss_fcn = torch.nn.CrossEntropyLoss()

        # Metrics
        def torch_acc(y_pred, y_true):
            train_acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
            return train_acc

        metrics = [torch_acc]

        training_params = {'epochs': epochs, 'optimizer': optimizer, 'loss': loss_fcn, 'metrics': metrics}

        return model, training_params
    
    elif dataset_name == 'Malaria' or dataset_name == 'Malaria_noaug':
        if model_name == 'E2_C4':
            
            """ Generate the model """

            # 115.067 parameters

            scale = 0.725
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=4)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 64, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=4, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x

        elif model_name == 'E2_C8':

            """ Generate the model """

            # 115.690 parameters

            scale = 0.525
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=8)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 64, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=4, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'E2_C16':

            """ Generate the model """

            # 102.836 parameters

            scale = 0.370
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=16)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 64, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=4, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'small_SO2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        #torch.nn.BatchNorm1d(int(40*scale)),
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True),
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_cut':

            scale = 1.7

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='same', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_vanilla':

            scale = 0.9

            class CNN(torch.nn.Module):
    
                def __init__(self, n_classes=2):
                    
                    super(CNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=3, out_channels=int(8*scale), kernel_size=9, padding=4, bias=True),
                        torch.nn.BatchNorm2d(int(8*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(8*scale), out_channels=int(16*scale), kernel_size=7, padding=3, bias=True),
                        torch.nn.BatchNorm2d(int(16*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(16*scale), out_channels=int(24*scale), kernel_size=7, padding=3, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(24*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(32*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(32*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(32*scale), out_channels=int(40*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(40*scale)),
                        torch.nn.ReLU(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'large_vanilla':

            # Extracted from https://github.com/rasbt/deeplearning-models/

            import torch.nn.functional as F

            def conv3x3(in_planes, out_planes, stride=1):
                """3x3 convolution with padding"""
                return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)

            class BasicBlock(torch.nn.Module):
                expansion = 1

                def __init__(self, inplanes, planes, stride=1, downsample=None):
                    super(BasicBlock, self).__init__()
                    self.conv1 = conv3x3(inplanes, planes, stride)
                    self.bn1 = torch.nn.BatchNorm2d(planes)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.conv2 = conv3x3(planes, planes)
                    self.bn2 = torch.nn.BatchNorm2d(planes)
                    self.downsample = downsample
                    self.stride = stride

                def forward(self, x):
                    residual = x

                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)

                    out = self.conv2(out)
                    out = self.bn2(out)

                    if self.downsample is not None:
                        residual = self.downsample(x)

                    out += residual
                    out = self.relu(out)

                    return out

            class CNN(torch.nn.Module):

                def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, grayscale=False):
                    self.inplanes = 64
                    if grayscale:
                        in_dim = 1
                    else:
                        in_dim = 3
                    super(CNN, self).__init__()
                    self.conv1 = torch.nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
                    self.bn1 = torch.nn.BatchNorm2d(64)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    self.layer1 = self._make_layer(block, 64, layers[0])
                    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                    #self.avgpool = torch.nn.AvgPool2d(7, stride=1)
                    self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

                    for m in self.modules():
                        if isinstance(m, torch.nn.Conv2d):
                            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                            m.weight.data.normal_(0, (2. / n)**.5)
                        elif isinstance(m, torch.nn.BatchNorm2d):
                            m.weight.data.fill_(1)
                            m.bias.data.zero_()

                def _make_layer(self, block, planes, blocks, stride=1):
                    downsample = None
                    if stride != 1 or self.inplanes != planes * block.expansion:
                        downsample = torch.nn.Sequential(
                            torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                            torch.nn.BatchNorm2d(planes * block.expansion),
                        )

                    layers = []
                    layers.append(block(self.inplanes, planes, stride, downsample))
                    self.inplanes = planes * block.expansion
                    for i in range(1, blocks):
                        layers.append(block(self.inplanes, planes))

                    return torch.nn.Sequential(*layers)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)

                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    # because MNIST is already 1x1 here:
                    # disable avg pooling
                    x = torch.mean(x, axis=(2,3))
                    
                    x = x.view(x.size(0), -1)
                    logits = self.fc(x)
                    #probas = F.softmax(logits, dim=1)
                    return logits

        elif model_name == 'E2_HN_1' or model_name == 'E2_HN_3' or model_name == 'E2_irr_1'  or model_name == 'E2_irr_3' or model_name == 'E2_irr_1_flip' or model_name == 'E2_irr_3_flip':

            """ Generate the model """

            # 115.690 parameters

            from ExpE2 import ExpE2SFCNN

        else:
            raise Exception("Model not implemented")

        """ Training params """

        if model_name == 'E2_HN_1':
            model = ExpE2SFCNN(n_channels=3, n_classes=2, N=-1, layer_type="hnet_conv", flip=False, scale=1.33)
        elif model_name == 'E2_HN_3':
            model = ExpE2SFCNN(n_channels=3, n_classes=2, N=-3, layer_type="hnet_conv", flip=False, scale=0.66)
        elif model_name == 'E2_irr_1':
            model = ExpE2SFCNN(n_channels=3, n_classes=2, N=-1, layer_type="realhnet", flip=False, scale=1.45)
        elif model_name == 'E2_irr_3':
            model = ExpE2SFCNN(n_channels=3, n_classes=2, N=-3, layer_type="realhnet", flip=False, scale=0.85)
        elif model_name == 'E2_irr_1_flip':
            model = ExpE2SFCNN(n_channels=3, n_classes=2, N=-1, layer_type="realhnet", flip=True, scale=1.1)
        elif model_name == 'E2_irr_3_flip':
            model = ExpE2SFCNN(n_channels=3, n_classes=2, N=-3, layer_type="realhnet", flip=True, scale=0.65)
        elif 'O2' in model_name:
            model = BCNN()
        elif 'vanilla' in model_name:
            model = CNN()
        else:
            model = E2CNN()

        # Epochs
        epochs = 50

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Loss function
        loss_fcn = torch.nn.CrossEntropyLoss()

        # Metrics
        def torch_acc(y_pred, y_true):
            train_acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
            return train_acc

        metrics = [torch_acc]

        training_params = {'epochs': epochs, 'optimizer': optimizer, 'loss': loss_fcn, 'metrics': metrics}

        return model, training_params
    
    elif dataset_name == 'bigearthnet' or dataset_name == 'bigearthnet_noaug':
        if model_name == 'E2_C4':
            
            """ Generate the model """

            # 115.067 parameters

            scale = 0.725
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=4)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 120, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x

        elif model_name == 'E2_C8':

            """ Generate the model """

            # 115.690 parameters

            scale = 0.525
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=8)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 120, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'E2_C16':

            """ Generate the model """

            # 102.836 parameters

            scale = 0.370
            
            class E2CNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(E2CNN, self).__init__()
                    
                    self.r2_act = gspaces.Rot2dOnR2(N=16)

                    in_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
                    self.input_type = in_type
                    
                    #
                    # Block 1
                    #

                    # convolution 1
                    out_type = nn.FieldType(self.r2_act, int(8*scale)*[self.r2_act.regular_repr])
                    self.block1 = nn.SequentialModule(
                        nn.MaskModule(in_type, 120, margin=1),
                        nn.R2Conv(in_type, out_type, kernel_size=9, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 2
                    in_type = self.block1.out_type
                    out_type = nn.FieldType(self.r2_act, int(16*scale)*[self.r2_act.regular_repr])
                    self.block2 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool1 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 2
                    #
                    
                    # convolution 3
                    in_type = self.block2.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block3 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 4
                    in_type = self.block3.out_type
                    out_type = nn.FieldType(self.r2_act, int(24*scale)*[self.r2_act.regular_repr])
                    self.block4 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )

                    self.pool2 = nn.SequentialModule(
                        nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                    )

                    #
                    # Block 3
                    #
                    
                    # convolution 5
                    in_type = self.block4.out_type
                    out_type = nn.FieldType(self.r2_act, int(32*scale)*[self.r2_act.regular_repr])
                    self.block5 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    # convolution 6
                    in_type = self.block5.out_type
                    out_type = nn.FieldType(self.r2_act, int(40*scale)*[self.r2_act.regular_repr])
                    self.block6 = nn.SequentialModule(
                        nn.R2Conv(in_type, out_type, kernel_size=7, padding=0, bias=False),
                        nn.InnerBatchNorm(out_type),
                        nn.ReLU(out_type, inplace=True)
                    )
                    
                    #
                    # Final layers
                    #

                    self.gpool = nn.GroupPooling(out_type)
                    
                    # number of output channels
                    c = self.gpool.out_type.size
                    
                    # Fully Connected
                    self.fully_net = torch.nn.Sequential(
                        torch.nn.Linear(c, n_classes),
                    )
                
                def forward(self, input: torch.Tensor):
                    # wrap the input tensor in a GeometricTensor
                    # (associate it with the input type)
                    x = nn.GeometricTensor(input, self.input_type)
                    
                    # apply each equivariant block
                    
                    # Each layer has an input and an output type
                    # A layer takes a GeometricTensor in input.
                    # This tensor needs to be associated with the same representation of the layer's input type
                    #
                    # The Layer outputs a new GeometricTensor, associated with the layer's output type.
                    # As a result, consecutive layers need to have matching input/output types
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.pool1(x)
                    
                    x = self.block3(x)
                    x = self.block4(x)
                    x = self.pool2(x)
                    
                    x = self.block5(x)
                    x = self.block6(x)
                    
                    # pool over the group
                    x = self.gpool(x)

                    # unwrap the output GeometricTensor
                    # (take the Pytorch tensor and discard the associated representation)
                    x = x.tensor
                    
                    # classify with the final fully connected layers)
                    x = torch.sum(x, dim=[2,3]) / (x.shape[2]**2)
                    x = self.fully_net(x.reshape(x.shape[0], -1))
                    
                    return x
                
        elif model_name == 'small_SO2':

            scale = 2.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_cut':

            scale = 1.7

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=False, scale_inv=False, cutoff='strong', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(8*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(16*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(24*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(32*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True,
                                    reflex_inv=True, scale_inv=False, cutoff='soft', TensorCorePad=False),
                        #AttentiveNorm2d(int(40*scale), n_mixtures=5, eps=0.001, momentum=0.99),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_SO2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=False, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_O2_plus':

            scale = 1.

            class BCNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(BCNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        BesselConv2d(C_in=3, C_out=int(8*scale), k=9, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(8*scale), C_out=int(16*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(16*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(16*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(24*scale), C_out=int(24*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),

                        GaussianBlur2d(C_in=int(24*scale), sigma=0.66),
                        torch.nn.AvgPool2d(2),

                        BesselConv2d(C_in=int(24*scale), C_out=int(32*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                        BesselConv2d(C_in=int(32*scale), C_out=int(40*scale), k=7, padding='valid', bias=True, scales=[-2,0,2],
                                    reflex_inv=True, scale_inv=True, cutoff='soft', TensorCorePad=False),
                        torch.nn.Tanh(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'small_vanilla':

            scale = 0.9

            class CNN(torch.nn.Module):
    
                def __init__(self, n_classes=43):
                    
                    super(CNN, self).__init__()

                    self.cnn = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=3, out_channels=int(8*scale), kernel_size=9, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(8*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(8*scale), out_channels=int(16*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(16*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(16*scale), out_channels=int(24*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(24*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(24*scale)),
                        torch.nn.ReLU(),

                        torch.nn.AvgPool2d(2),

                        torch.nn.Conv2d(in_channels=int(24*scale), out_channels=int(32*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(32*scale)),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(in_channels=int(32*scale), out_channels=int(40*scale), kernel_size=7, padding=0, bias=True),
                        torch.nn.BatchNorm2d(int(40*scale)),
                        torch.nn.ReLU(),
                    )

                    self.dense = torch.nn.Sequential(
                        torch.nn.Linear(in_features=int(40*scale), out_features=n_classes, bias=True)
                    )

                def forward(self, input: torch.Tensor):

                    x = self.cnn(input)
                    x = torch.mean(x, axis=(2,3))
                    x = self.dense(x)

                    return x
                
        elif model_name == 'large_vanilla':

            # Extracted from https://github.com/rasbt/deeplearning-models/

            import torch.nn.functional as F

            def conv3x3(in_planes, out_planes, stride=1):
                """3x3 convolution with padding"""
                return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)

            class BasicBlock(torch.nn.Module):
                expansion = 1

                def __init__(self, inplanes, planes, stride=1, downsample=None):
                    super(BasicBlock, self).__init__()
                    self.conv1 = conv3x3(inplanes, planes, stride)
                    self.bn1 = torch.nn.BatchNorm2d(planes)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.conv2 = conv3x3(planes, planes)
                    self.bn2 = torch.nn.BatchNorm2d(planes)
                    self.downsample = downsample
                    self.stride = stride

                def forward(self, x):
                    residual = x

                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)

                    out = self.conv2(out)
                    out = self.bn2(out)

                    if self.downsample is not None:
                        residual = self.downsample(x)

                    out += residual
                    out = self.relu(out)

                    return out

            class CNN(torch.nn.Module):

                def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=43, grayscale=False):
                    self.inplanes = 64
                    if grayscale:
                        in_dim = 1
                    else:
                        in_dim = 3
                    super(CNN, self).__init__()
                    self.conv1 = torch.nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
                    self.bn1 = torch.nn.BatchNorm2d(64)
                    self.relu = torch.nn.ReLU(inplace=True)
                    self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    self.layer1 = self._make_layer(block, 64, layers[0])
                    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                    #self.avgpool = torch.nn.AvgPool2d(7, stride=1)
                    self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

                    for m in self.modules():
                        if isinstance(m, torch.nn.Conv2d):
                            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                            m.weight.data.normal_(0, (2. / n)**.5)
                        elif isinstance(m, torch.nn.BatchNorm2d):
                            m.weight.data.fill_(1)
                            m.bias.data.zero_()

                def _make_layer(self, block, planes, blocks, stride=1):
                    downsample = None
                    if stride != 1 or self.inplanes != planes * block.expansion:
                        downsample = torch.nn.Sequential(
                            torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                            torch.nn.BatchNorm2d(planes * block.expansion),
                        )

                    layers = []
                    layers.append(block(self.inplanes, planes, stride, downsample))
                    self.inplanes = planes * block.expansion
                    for i in range(1, blocks):
                        layers.append(block(self.inplanes, planes))

                    return torch.nn.Sequential(*layers)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)

                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    # because MNIST is already 1x1 here:
                    # disable avg pooling
                    x = torch.mean(x, axis=(2,3))
                    
                    x = x.view(x.size(0), -1)
                    logits = self.fc(x)
                    #probas = F.softmax(logits, dim=1)
                    return logits

        elif model_name == 'E2_HN_1' or model_name == 'E2_HN_3' or model_name == 'E2_irr_1'  or model_name == 'E2_irr_3' or model_name == 'E2_irr_1_flip' or model_name == 'E2_irr_3_flip':

            """ Generate the model """

            # 115.690 parameters

            from ExpE2 import ExpE2SFCNN

        else:
            raise Exception("Model not implemented")

        """ Training params """

        if model_name == 'E2_HN_1':
            model = ExpE2SFCNN(n_channels=3, n_classes=43, N=-1, layer_type="hnet_conv", flip=False, scale=1.33)
        elif model_name == 'E2_HN_3':
            model = ExpE2SFCNN(n_channels=3, n_classes=43, N=-3, layer_type="hnet_conv", flip=False, scale=0.66)
        elif model_name == 'E2_irr_1':
            model = ExpE2SFCNN(n_channels=3, n_classes=43, N=-1, layer_type="realhnet", flip=False, scale=1.45)
        elif model_name == 'E2_irr_3':
            model = ExpE2SFCNN(n_channels=3, n_classes=43, N=-3, layer_type="realhnet", flip=False, scale=0.85)
        elif model_name == 'E2_irr_1_flip':
            model = ExpE2SFCNN(n_channels=3, n_classes=43, N=-1, layer_type="realhnet", flip=True, scale=1.1)
        elif model_name == 'E2_irr_3_flip':
            model = ExpE2SFCNN(n_channels=3, n_classes=43, N=-3, layer_type="realhnet", flip=True, scale=0.65)
        elif 'O2' in model_name:
            model = BCNN()
        elif 'vanilla' in model_name:
            model = CNN()
        else:
            model = E2CNN()

        # Epochs
        epochs = 50

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Loss function
        loss_fcn = torch.nn.CrossEntropyLoss()

        # Metrics
        from torcheval.metrics import MultilabelAccuracy

        def mutli_label_exact(y_pred, y_true):
            metric = MultilabelAccuracy(criteria="exact_match")
            metric = metric.update(torch.nn.functional.sigmoid(y_pred) > 0.5, y_true)
            train_acc = metric.compute().float().mean()
            return train_acc
        
        def mutli_label_hamming(y_pred, y_true):
            metric = MultilabelAccuracy(criteria="hamming")
            metric = metric.update(torch.nn.functional.sigmoid(y_pred) > 0.5, y_true)
            train_acc = metric.compute().float().mean()
            return train_acc

        metrics = [mutli_label_exact, mutli_label_hamming]

        training_params = {'epochs': epochs, 'optimizer': optimizer, 'loss': loss_fcn, 'metrics': metrics}

        return model, training_params

    else:
        raise Exception("Dataset not implemented")
