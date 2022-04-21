import torch
import torch.nn.functional as F


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def normalize(A, symmetric=True):
    A = A + torch.eye(A.size(0))
    d = A.sum(1)
    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, 240, bias=False)
        self.fc3 = torch.nn.Linear(240, dim_out, bias=False)

    def forward(self, A, X):
        X = F.relu(self.fc1(torch.matmul(A, X)))
        return self.fc3(torch.matmul(A, X))


class DCGNNEncoder(torch.nn.Module):

    def __init__(self, Adj, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(DCGNNEncoder, self).__init__()
        self.adj = Adj
        self.causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        self.reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        self.squeeze = SqueezeChannels()
        self.gcn = GCN(reduced_size, out_channels)
        self.linear = torch.nn.Linear(reduced_size, out_channels)

    def forward(self, x, epoch, current_epoch):
        res = self.causal_cnn(x)
        res = self.reduce_size(res)
        res = self.squeeze(res)
        res = self.gcn(self.adj, res)
        return res

