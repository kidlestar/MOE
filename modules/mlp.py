# -*- coding: utf-8 -*-
from torch.nn import functional as F
from modules.dropout import SharedDropout, SharedDropout_convex, SharedDropout_inf, MSharedDropout_
import torch
import torch.nn as nn
import math
from torch.nn.utils import spectral_norm
from torch.nn.functional import normalize
def reshape_weight_to_matrix(weight):
    weight_mat = weight
    """
    if self.dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(self.dim,
                                        *[d for d in range(weight_mat.dim()) if d != self.dim])
    """
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)

class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
class CMLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0):

        super(CMLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout_inf(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class MMLP_(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0):

        super(MMLP_, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = MSharedDropout_(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, mask=None):
        x = self.linear(x)
        x = self.activation(x)
        x, mask = self.dropout(x, mask)

        return x, mask




class MLP_static(nn.Module):

    def __init__(self, n_in, n_hidden):
        super(MLP_static, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)

        return x




class MLP_const(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP_const, self).__init__()

        self.linear = spectral_norm(nn.Linear(n_in, n_hidden))

        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_normal_(self.linear.weight)
        #nn.init.kaiming_normal_(self.linear.weight,mode='fan_in')
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        return x

    def evaluate(self, x, training):
        device = self.linear.bias.device
        self.eps = 1e-12
        with torch.no_grad():
            weight = self.linear.weight_orig.detach().to(device)
            u = self.linear.weight_u.detach().to(device)
            v = self.linear.weight_v.detach().to(device)
            weight_mat = reshape_weight_to_matrix(weight)
            if training:
                v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
            #u = u.clone(memory_format=torch.contiguous_format)
            #v = v.clone(memory_format=torch.contiguous_format)
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            weight = weight / sigma
        x=F.linear(x, weight, self.linear.bias.detach())
        return x


class MLP_convex(nn.Module):

    def __init__(self, n_in, n_hidden, dropout):
        super(MLP_convex, self).__init__()

        self.w = nn.Parameter(torch.randn(n_in,n_hidden))
        self.b = nn.Parameter(torch.randn(n_hidden))
        self.dropout = SharedDropout_convex(p=dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        #self.activation = nn.CELU()

        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_normal_(self.w)
        #nn.init.kaiming_uniform_(self.w,a=math.sqrt(5))
        nn.init.orthogonal_(self.w)
        nn.init.zeros_(self.b)
        #fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.w)
        #bound = 1 / math.sqrt(fan_out)
        #nn.init.uniform_(self.b, -bound, bound)

        #init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #if self.bias is not None:
            #fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1 / math.sqrt(fan_in)
            #init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        x = self.activation(torch.matmul(x,torch.abs(self.w)) + self.b)
        x = self.dropout(x)
        return x

    def evaluate(self,x):
        x = self.activation(torch.matmul(x,torch.abs(self.w.detach())) + self.b.detach())
        x = self.dropout(x)
        return x

    def set(self,mask=None):
        self.dropout.set(mask)


class MLP_convex_const(nn.Module):

    def __init__(self, n_in, n_hidden, dropout):
        super(MLP_convex_const, self).__init__()

        self.w = nn.Parameter(torch.randn(n_in,n_hidden))
        self.b = nn.Parameter(torch.randn(n_hidden))
        #self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout_convex(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_normal_(self.w)
        #nn.init.kaiming_uniform_(self.w,a=math.sqrt(5))
        nn.init.orthogonal_(self.w)
        nn.init.zeros_(self.b)
        #fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.w)
        #bound = 1 / math.sqrt(fan_out)
        #nn.init.uniform_(self.b, -bound, bound) 

    def forward(self,x):
        x = self.dropout(torch.matmul(x,torch.abs(self.w)) + self.b)
        return x

    def evaluate(self,x):
        x = self.dropout(torch.matmul(x,torch.abs(self.w.detach())) + self.b.detach())
        return x

    def set(self, mask=None):
        self.dropout.set(mask)



class MLP_convex_static(nn.Module):

    def __init__(self, n_in, n_hidden):
        super(MLP_convex_static, self).__init__()

        self.w = nn.Parameter(torch.randn(n_in,n_hidden))
        self.b = nn.Parameter(torch.randn(n_hidden))
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        #self.activation = nn.CELU()
        #self.dropout = SharedDropout_convex(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_normal_(self.w)
        #nn.init.kaiming_uniform_(self.w,a=0.1,mode='fan_out',nonlinearity='leaky_relu')
        nn.init.orthogonal_(self.w)
        nn.init.zeros_(self.b)
        #fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.w)
        #bound = 1 / math.sqrt(fan_out)
        #nn.init.uniform_(self.b, -bound, bound) 

    def forward(self,x):
        x = self.activation(torch.matmul(x,torch.abs(self.w)) + self.b)
        return x

    def evaluate(self,x):
        x = self.activation(torch.matmul(x,torch.abs(self.w.detach())) + self.b.detach())
        return x

class MLP_convex_const_static(nn.Module):

    def __init__(self, n_in, n_hidden):
        super(MLP_convex_const_static, self).__init__()

        self.w = nn.Parameter(torch.randn(n_in,n_hidden))
        self.b = nn.Parameter(torch.randn(n_hidden))
        #self.activation = nn.LeakyReLU(negative_slope=0.1)
        #self.dropout = SharedDropout_convex(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_normal_(self.w)
        #nn.init.kaiming_uniform_(self.w,a=math.sqrt(5))
        nn.init.orthogonal_(self.w)
        nn.init.zeros_(self.b)
        #fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.w)
        #bound = 1 / math.sqrt(fan_out)
        #nn.init.uniform_(self.b, -bound, bound) 

    def forward(self,x):
        x = torch.matmul(x,torch.abs(self.w)) + self.b
        return x

    def evaluate(self,x):
        x = torch.matmul(x,torch.abs(self.w.detach())) + self.b.detach()
        return x




