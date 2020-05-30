from cnn_gp import Conv2d, ReLU, Sequential, resnet_block

def CNTK_nopool(channels=None, depth=14):
    return Sequential(*([Conv2d(kernel_size=3), ReLU()]*depth))


def PreResNetNoPooling(depth, channels=None):
    assert (depth - 2) % 6 == 0, "depth should be 6n+2"
    num_blocks_per_block = (depth - 2) // 6
    return Sequential(
        Conv2d(kernel_size=3),

        resnet_block(stride=1, projection_shortcut=True, multiplier=1),
        *([resnet_block(stride=1, projection_shortcut=False, multiplier=1)]*(num_blocks_per_block-1)),

        resnet_block(stride=2, projection_shortcut=True, multiplier=2),
        *([resnet_block(stride=1, projection_shortcut=False, multiplier=2)]*(num_blocks_per_block-1)),

        resnet_block(stride=2, projection_shortcut=True, multiplier=4),
        *([resnet_block(stride=1, projection_shortcut=False, multiplier=4)]*(num_blocks_per_block-1)),
    )
