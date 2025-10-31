import torch
import torch.nn as nn

'''Deep Q-Network'''
class DQN(nn.Module):
    def __init__(self, input_shape: tuple, action_space: int):
        super(DQN, self).__init__()
        conv_channels = [32, 64, 64]
        conv_kernel_size = [8, 4, 3]
        conv_stride = [4, 2, 1]
        fc1_output_features = [512]

        # 卷积层
        self.conv1 = nn.Conv2d(input_shape[0], conv_channels[0], kernel_size=conv_kernel_size[0], stride=conv_stride[0], bias=False)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=conv_kernel_size[1], stride=conv_stride[1], bias=False)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=conv_kernel_size[2], stride=conv_stride[2], bias=False)

        # 计算经过卷积层后的输出尺寸
        conv1_output_shape = tuple((i - conv_kernel_size[0]) // conv_stride[0] + 1 for i in input_shape[1:])
        conv2_output_shape = tuple((i - conv_kernel_size[1]) // conv_stride[1] + 1 for i in conv1_output_shape)
        conv3_output_shape = tuple((i - conv_kernel_size[2]) // conv_stride[2] + 1 for i in conv2_output_shape)

        # 全连接层
        self.fc1 = nn.Linear(conv_channels[2] * conv3_output_shape[0] * conv3_output_shape[1], fc1_output_features[0])
        self.fc2 = nn.Linear(fc1_output_features[0], action_space)

    def forward(self, x):
        x = x / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
'''Dueling Deep Q-Network'''
class DuelingDQN(nn.Module):
    def __init__(self, input_shape: tuple, action_space: int):
        super(DuelingDQN, self).__init__()
        conv_channels = [32, 64, 64]
        conv_kernel_size = [8, 4, 3]
        conv_stride = [4, 2, 1]
        fc_output_features = 512

        # 卷积层
        self.conv1 = nn.Conv2d(input_shape[0], conv_channels[0], kernel_size=conv_kernel_size[0], stride=conv_stride[0], bias=False)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=conv_kernel_size[1], stride=conv_stride[1], bias=False)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=conv_kernel_size[2], stride=conv_stride[2], bias=False)

        # 计算经过卷积层后的输出尺寸
        conv1_output_shape = tuple((i - conv_kernel_size[0]) // conv_stride[0] + 1 for i in input_shape[1:])
        conv2_output_shape = tuple((i - conv_kernel_size[1]) // conv_stride[1] + 1 for i in conv1_output_shape)
        conv3_output_shape = tuple((i - conv_kernel_size[2]) // conv_stride[2] + 1 for i in conv2_output_shape)

        # 计算卷积输出尺寸
        flattened_size = conv_channels[2] * conv3_output_shape[0] * conv3_output_shape[1]

        self.vfc1 = nn.Linear(flattened_size, fc_output_features)
        self.afc1 = nn.Linear(flattened_size, fc_output_features)

        self.vfc2 = nn.Linear(fc_output_features, 1)
        self.afc2 = nn.Linear(fc_output_features, action_space)

    def forward(self, x) -> torch.Tensor:
        x = x / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        v = torch.relu(self.vfc1(x))
        a = torch.relu(self.afc1(x))

        v = self.vfc2(v)
        a = self.afc2(a)

        return v + a - a.mean(1, keepdim=True)