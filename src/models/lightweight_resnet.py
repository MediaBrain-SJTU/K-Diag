from torch import nn

class ResNet(nn.Module):

    def __init__(self,
                 num_classes=10,
                 proj_type='linear',
                 num_blocks=[1, 1, 1, 1],
                 c_hidden=[16, 32, 64, 128],
                 init_max_pool=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.proj_type = proj_type
        self.num_blocks = num_blocks
        self.c_hidden = c_hidden
        self.init_max_pool = init_max_pool
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.c_hidden

        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, stride=2,
                      bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            nn.ReLU(),
        )
        if self.init_max_pool:
            self.input_net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        blocks = []
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                blocks.append(
                    ResNetBlock(c_in=c_hidden[
                        block_idx if not subsample else (block_idx - 1)],
                                subsample=subsample,
                                c_out=c_hidden[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(self, c_in, subsample=False, c_out=-1):
        super().__init__()
        if not subsample:
            c_out = c_in

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                      stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1,
                                    stride=2) if subsample else None
        self.act_fn = nn.ReLU()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out
    
import torch
import torchvision
  
def pgn_switch(
        model_type,
        out_features,
        **kwargs
) -> torch.nn.Module:
    if model_type == 'resnet18':
        return generate_resnet18(out_features, **kwargs)
    elif model_type == 'resnet10':
        return create_small_resnet(out_features, **kwargs)


def generate_resnet18(out_features, pretrained_pgn, **kwargs):
    resnet = torchvision.models.resnet18(pretrained=pretrained_pgn)
    resnet.fc = torch.nn.Linear(
        in_features=512,
        out_features=out_features
    )
    return resnet


def create_small_resnet(out_features,
                        proj_type,
                        blocks_per_group,
                        initial_channels,
                        nr_groups,
                        init_max_pool=False,
                        **kwargs):
    resnet = ResNet(
        num_classes=out_features,
        proj_type=proj_type,
        num_blocks=[blocks_per_group] * nr_groups,
        c_hidden=[initial_channels * (2 ** power) for power in range(nr_groups)],
        init_max_pool=init_max_pool,
    )
    return resnet