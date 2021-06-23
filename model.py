import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class cosAtt(nn.Module):
    def __init__(self, device, num_nodes):
        super(cosAtt, self).__init__()
        self.att_beta = nn.Parameter(torch.randn(1, 1, num_nodes, num_nodes).to(device), requires_grad=True).to(device)
    
    def forward(self, x):
        norm2 = torch.norm(x, dim=-1, keepdim=True)
        x_result = torch.matmul(x, torch.transpose(x, 2, 3))
        norm2_result = torch.matmul(norm2, torch.transpose(norm2, 2, 3))
        

        cos = torch.div(x_result, norm2_result+1e-7)
        cos_ = self.att_beta * cos
        P = torch.sigmoid(cos_)
        output = torch.matmul(P, x)
        return output

class gcn_att(nn.Module):
    def __init__(self,device, num_nodes, A, c_in, c_out):
        super(gcn_att,self).__init__()
        L = util.scaled_laplacian(A)
        self.Ks = 3
        self.Lk = util.cheb_poly_approx(L, self.Ks, A.size()[0])
        self.cosAtt = cosAtt(device, num_nodes)
        self.theta = nn.Parameter(torch.randn(self.Ks*c_in, c_out).to(device), requires_grad=True).to(device)
        self.bias = nn.Parameter(torch.zeros(c_out).to(device), requires_grad=True).to(device)

    def forward(self, x, c_in, c_out):
        _, _, _, time_steps = x.size()
        kernel = torch.Tensor(self.Lk).to('cuda:0')
        n = kernel.shape[0]
        x_input = torch.reshape(x, [-1, n, c_in])
        x_tmp = torch.reshape(torch.transpose(x_input, 1, 2), [-1, n])
        x_mul = torch.reshape(torch.matmul(x_tmp, kernel), [-1, c_in, self.Ks, n])
        x_ker = torch.reshape(torch.transpose(torch.transpose(x_mul, 1, 2), 1, 3), [-1, c_in * self.Ks])
        x_gconv = torch.reshape(torch.matmul(x_ker, self.theta),[-1, n, c_out])+self.bias
        x_gc = torch.reshape(x_gconv, [-1, time_steps, n, c_out])

        cos = self.cosAtt(x)
        x = cos * torch.relu(torch.transpose(x_gc[:, :, :, 0:c_out], 1, 3) + x)
        return F.softmax(x, dim=1)

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class glu(nn.Module):
    def __init__(self, Kt, c_in, c_out, device):
        super(glu, self).__init__()
        # self.wt = nn.Parameter(torch.randn(Kt, 1, c_in, c_out*2).to(device), requires_grad=True).to(device)
        # self.bt = nn.Parameter(torch.zeros(c_out*2).to(device), requires_grad=True).to(device)
        self.conv = nn.Conv1d(in_channels=c_in,
                                out_channels=c_out*2,
                                kernel_size=(1, Kt))
        self.c_in = c_in
        self.c_out = c_out
        self.Kt = Kt

    def forward(self, x, device):
        batch_size, _, n, T = x.size()
        if self.c_in > self.c_out:
            w_input = nn.Parameter(torch.randn(1, 1, self.c_in, self.c_out).to(device), requires_grad=True).to(device)
            x_input = torch.conv1d(x, w_input, stride=[1,1,1,1], padding='SAME')
        elif self.c_in < self.c_out:
            x_input = torch.cat((x, torch.zeros(batch_size, self.c_out-self.c_in, n, T).to(device)), 1)
        else:
            x_input = x

        x_input_tmp = x_input[:, :, :, self.Kt-1:T]

        x_conv = self.conv(x)
        # x_conv = torch.conv1d(x_input, self.wt, self.bt, padding=0, stride=1)

        return (x_conv[:, 0:self.c_out, :, :] + x_input_tmp) * torch.sigmoid(x_conv[:, -self.c_out:, :, :])


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=1,residual_channels=32,dilation_channels=32,skip_channels=32,end_channels=512,kernel_size=3,layers=12, step=1):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.step = step
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.first_glu = nn.ModuleList()
        # self.second_glu = nn.ModuleList()
        self.third_glu = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        # self.conv = nn.ModuleList() # new 0323


        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(layers):
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size)))

            self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size)))
            # self.conv.append(nn.Conv1d(in_channels=residual_channels, # new
            #                                      out_channels=dilation_channels,
            #                                      kernel_size=(1, 10)))
            # self.first_glu.append(glu(kernel_size, residual_channels, dilation_channels, device))
            # self.second_glu.append(glu(kernel_size, residual_channels, dilation_channels, device))
            self.third_glu.append(glu(12-kernel_size+1-self.step+1, dilation_channels, dilation_channels, device))
            
            # 1x1 convolution for residual connection
            self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))

            # 1x1 convolution for skip connection
            # self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
            #                                     out_channels=skip_channels,
            #                                     kernel_size=(1, 1)))

            self.bn.append(nn.BatchNorm2d(residual_channels))
            # new_dilation *=2
            # receptive_field += additional_scope
            # additional_scope *= 2
            if self.gcn_bool:
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input, device):
        # in_len = input.size(3)
        # if in_len<self.receptive_field:
        #     x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        # else:
        #     x = input
        x = input # 64, 2, 170, 12
        x = self.start_conv(x) # 64, 32, 170, 12
        skip = []

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x # 62, 32, 170, 12

            # dilated convolution
            filter = self.filter_convs[i](residual)
            # gate = torch.sigmoid(filter) # 一会删
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # x = self.first_glu[i](residual, device)
            
            # x_1 = (x[:, 0:32, :, :] + residual[:, :, :, 2:12]) * torch.sigmoid(x[:, -32:, :, :])
            # x = (x_conv[:, 0:self.c_out, :, :] + x_input_tmp) * torch.sigmoid(x_conv[:, -self.c_out:, :, :])

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x_gcn = self.gconv[i](x, new_supports)
                else:
                    x_gcn = self.gconv[i](x,self.supports)
            else:
                x_gcn = self.residual_convs[i](x)

            # x_2 = self.second_glu[i](x_gcn,device)
            # # x_2 = (_x_2[:, 0:32, :, :] + residual[:, :, :, 2:10]) * torch.sigmoid(_x_2[:, -32:, :, :])
            

            # x_2_norm = self.bn[i](x_2)

            _, channel, n, _ = x_gcn.size() #64 32 170 10

            x_3 = self.third_glu[i](x_gcn,device) #64 32 170 1
            # x_3 = self.conv[i](x_gcn)

            # x_3 = (_x_3[:, 0:32, :, :] + residual[:, :, :, 11:12]) * torch.sigmoid(_x_3[:, -32:, :, :])
            

            x_3_norm = self.bn[i](x_3)

            tmp_res = residual[:, :, :, self.step:12]
            residual = torch.cat((tmp_res, x_3_norm),3)
            # residual = torch.cat((tmp_res, x_3),3)
            # residual[:, :, :, 0:self.layers-1] = residual[:, :, :, 1:self.layers]
            # residual[:, :, :, self.layers-1:] = x_3_norm

            
            skip.append(x_3_norm)
            # skip.append(x_3)

            x = residual

        skip_1 = torch.cat(skip, dim=3)
        # skip = torch.transpose(skip, 1, 3)

        # x = F.relu(skip) # 64, 256, 170, 1
        x = F.relu(self.end_conv_1(skip_1))
        x1 = self.end_conv_2(x) # 64, 12, 170, 1
        return torch.transpose(x1, 1, 3)
