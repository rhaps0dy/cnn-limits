from cnn_gp import Conv2d, ReLU, Sequential, resnet_block

def CNTK_nopool(channels=None, depth=14):
    return Sequential(*([Conv2d(kernel_size=3), ReLU()]*depth))
