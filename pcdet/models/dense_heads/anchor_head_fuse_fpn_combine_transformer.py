import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy.spatial.distance import cdist

from .anchor_head_template import AnchorHeadTemplate


import copy
from typing import Optional, Any

from torch import Tensor
from torch.nn import Module, MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm
from torch.nn.init import xavier_uniform_

class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(Transformer, self).__init__()
        self.self_attn = Self_Attn(d_model)

        # if custom_encoder is not None:
        #     self.encoder = custom_encoder
        # else:
        #     encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        #     encoder_norm = LayerNorm(d_model)
        #     self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # if custom_decoder is not None:
        #     self.decoder = custom_decoder
        # else:
        #     decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        #     decoder_norm = LayerNorm(d_model)
        #     self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src):
        src = self.self_attn(src)
        # print("src", src[0].shape)
        return src[0]

class Transformer2(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//32 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//32 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class LocationAttentionLayer(nn.Module):

    def __init__(self, num_channels, kernel_size, kernel_size2=0, no_sigmoid=False, detach=False):
        super(LocationAttentionLayer, self).__init__()
        if kernel_size2 == 0:
            kernel_size2 = kernel_size

        self.kernel_size = kernel_size
        self.kernel_size2 = kernel_size2
        self.patch_matrix = nn.Parameter(torch.randn(1, kernel_size, kernel_size2), requires_grad=True)
        # self.patch_conv = nn.Conv2d(num_channels, 1, kernel_size, kernel_size) n, 126, 126
        self.sigmoid = nn.Sigmoid()
        self.no_sigmoid = no_sigmoid
        self.detach = detach

    def forward(self, input_tensor):
        #2, 512, 126, 126

        # print("kernel_size", self.kernel_size, self.kernel_size2)
        # print("input_tensor", input_tensor.shape)
        bt, c, h, w = input_tensor.size()

        # print("bt, c, h, w", bt, c, h, w)
        # print("input_tensor", input_tensor.shape)
        # patch_tensor = self.patch_conv(input_tensor)
        # print("patch_tensor", patch_tensor.shape)
        # print("self.patch_matrix.repeat(bt*c, 1, 1)", self.patch_matrix.repeat(bt*c, 1, 1).shape)
        if self.no_sigmoid:
            input_tensor = input_tensor.contiguous().view(-1, h, w) #
            input_tensor = input_tensor * self.patch_matrix.repeat(bt*c, 1, 1)
            input_tensor = input_tensor.view(bt, c, h, w)
        else:
            input_tensor = input_tensor.view(-1, h, w) #
            att_matrix = self.patch_matrix.repeat(bt*c, 1, 1)
            # if self.detach:
            #     att_matrix = att_matrix.detach()
            input_tensor = input_tensor * att_matrix

            # z = x * att_matrix.detach()
            # z = x.detach() * att_matrix
            input_tensor = self.sigmoid(input_tensor).view(bt, c, h, w)

        return input_tensor

class LocationAttentionDoubleLayer(nn.Module):

    def __init__(self, num_channels, kernel_size, kernel_size2=0, no_sigmoid=False):
        super(LocationAttentionDoubleLayer, self).__init__()
        if kernel_size2 == 0:
            kernel_size2 = kernel_size
        self.patch_matrix = nn.Parameter(torch.randn(1, kernel_size, kernel_size2), requires_grad=True)
        # self.patch_conv = nn.Conv2d(num_channels, 1, kernel_size, kernel_size) n, 126, 126
        self.sigmoid = nn.Sigmoid()
        self.no_sigmoid = no_sigmoid

    def forward(self, input_tensor, dom_atten):
        #2, 512, 126, 126
        # print("dom_atten", dom_atten.shape) # 3, 514, 128, 128
        # print("input_tensor", input_tensor.shape) # , , 128, 128
        bt, c, h, w = input_tensor.size()

        # print("bt, c, h, w", bt, c, h, w)
        # print("input_tensor", input_tensor.shape)
        # patch_tensor = self.patch_conv(input_tensor)
        # print("patch_tensor", patch_tensor.shape)
        if self.no_sigmoid:
            input_tensor = input_tensor.contiguous().view(-1, h, w) #
            dom_atten = dom_atten.contiguous().view(-1, h, w)
            max_att = torch.max(dom_atten, self.patch_matrix.repeat(bt*c, 1, 1))
            input_tensor = input_tensor * max_att
            input_tensor = input_tensor.view(bt, c, h, w)
        else:
            input_tensor = input_tensor.view(-1, h, w) #
            dom_atten = dom_atten.view(-1, h, w) #
            max_att = torch.max(dom_atten, self.patch_matrix.repeat(bt*c, 1, 1))
            input_tensor = input_tensor * max_att
            input_tensor = self.sigmoid(input_tensor).view(bt, c, h, w)

        return input_tensor

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        # print("input_tensor.size()", input_tensor.size()) #2, 512, 126, 126

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        # print("out.size()", out.size()) #2, 1, 126, 126

        squeeze_tensor = self.sigmoid(out)
        # print("squeeze_tensor.size()", squeeze_tensor.size()) # 2, 1, 126, 126

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        # print("squeeze_tensor 2.size()", squeeze_tensor.size()) # 2, 1, 126, 126
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # print("output_tensor 2.size()", output_tensor.size()) #2, 512, 126, 126
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size() #2, 512, 126, 126
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2) #2, 512, 126*126(1)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class LocalDomainClassifier(nn.Module):
    def __init__(self, input_channels=256, context=False):
        super(LocalDomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        # print("sef context", self.context)
        self._init_weights()
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            #m.bias.data.zero_()
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = self.conv3(x)
            return F.sigmoid(x),feat
        else:
            x = self.conv3(x)
            return F.sigmoid(x)

class AnchorHeadFuseFPNCombineTransformer(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, nusc=False, input_channels_fpn=None, num_fpn_up=0, num_fpn_down=0, num_fpn_downup=0,  fpn_layers=[], voxel_size=[0.1, 0.1, 0.2], nheads=2, num_encoder_layers=1, num_decoder_layers=1, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, nusc=nusc,
            num_fpn_up=num_fpn_up, num_fpn_down=num_fpn_down, num_fpn_downup=num_fpn_downup, fpn_layers=fpn_layers, voxel_size=voxel_size, **kwargs
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        #####################################

        self.transformer = Transformer(input_channels, nheads, num_encoder_layers, num_decoder_layers)

        # Transformer(
        #     d_model=input_channels,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     num_encoder_layers=6,
        #     num_decoder_layers=6,
        #     normalize_before=False,
        #     return_intermediate_dec=True,
        # )

        self.row_embed = nn.Parameter(torch.rand(128, input_channels // 2))
        self.col_embed = nn.Parameter(torch.rand(128, input_channels // 2))

        self.query_pos = nn.Parameter(torch.rand(100, input_channels))
        # self.query_embed = nn.Embedding(100, input_channels)

        self.voxel_det_seconv_attention = self.model_cfg.get('VOXEL_DET_SECONV_ATTENTION', False)
        self.voxel_det_se_attention = self.model_cfg.get('VOXEL_DET_SE_ATTENTION', False)
        self.voxel_det_patch_attention = self.model_cfg.get('VOXEL_DET_PATCH_ATTENTION', False)
        self.voxel_dom_seconv_attention = self.model_cfg.get('VOXEL_DOM_SECONV_ATTENTION', False)
        self.voxel_dom_se_attention = self.model_cfg.get('VOXEL_DOM_SE_ATTENTION', False)
        self.voxel_dom_patch_attention = self.model_cfg.get('VOXEL_DOM_PATCH_ATTENTION', False)
        self.joint_attention = self.model_cfg.get('VOXEL_DETDOM_JOINT_ATTENTION', False)
        self.dom_patch_first = self.model_cfg.get('DOM_PATCH_FIRST', False)
        self.no_sigmoid = self.model_cfg.get('NO_SIGMOID', False)

        if self.sep_two_dom or (self.double_pma and not self.joint_pma):
            self.input_channels_dom_sep = input_channels

        if self.range_guidance:
            if self.range_guidance_dom_only:
                self.input_channels = input_channels
                if self.range_guidance_dist:
                    self.input_channels_dom = input_channels + 1
                else:
                    self.input_channels_dom = input_channels + 2
            else:
                if self.range_guidance_dist:
                    self.input_channels = input_channels + 1
                else:
                    self.input_channels = input_channels + 2
                self.input_channels_dom = self.input_channels
        else:
            self.input_channels = input_channels
            self.input_channels_dom = input_channels

        if self.joint_two_dom:
            if self.dom_patch_first or self.patch_unplug_context:
                self.input_channels_dom_joint = input_channels
            else:
                if self.range_guidance_dist:
                    self.input_channels_dom_joint = input_channels + 1
                else:
                    self.input_channels_dom_joint = input_channels + 2

        if self.joint_pma:
            self.input_channels_dom_joint = input_channels + 2

        self.num_keypoints_range = self.model_cfg.get('NUM_KEYPOINTS_RANGE', {})
        self.range_keys = self.num_keypoints_range.keys()

        self.point_fc_range = nn.ModuleDict()
        # self.domain_classifier_range = nn.ModuleDict()

        for i in self.range_keys:
            self.point_fc_range[i] = nn.Sequential(nn.Linear(self.num_keypoints_range[i], input_channels), nn.ReLU(True), nn.Dropout())

        self.input_channels_fpn = input_channels_fpn
        self.input_channels_dom_fpn = {}
        if self.sep_two_dom or (self.double_pma and not self.joint_pma):
            self.input_channels_dom_sep_fpn = {}
        if self.joint_two_dom or self.joint_pma:
            self.input_channels_dom_joint_fpn = {}

        for layer in self.fpn_layers:

            if self.sep_two_dom or (self.double_pma and not self.joint_pma):
                self.input_channels_dom_sep_fpn[layer] = input_channels_fpn[layer]

            if self.range_guidance:
                if self.range_guidance_dom_only:
                    if self.range_guidance_dist:
                        self.input_channels_dom_fpn[layer] = input_channels_fpn[layer]  + 1
                    else:
                        self.input_channels_dom_fpn[layer] = input_channels_fpn[layer]  + 2
                else:
                    if self.range_guidance_dist:
                        self.input_channels_fpn[layer] = input_channels_fpn[layer] + 1
                    else:
                        self.input_channels_fpn[layer] = input_channels_fpn[layer] + 2

                    self.input_channels_dom_fpn[layer] = self.input_channels_fpn[layer]
            else:
                self.input_channels_dom_fpn[layer] = input_channels_fpn[layer]

            if self.joint_two_dom:
                if self.dom_patch_first or self.patch_unplug_context:
                    self.input_channels_dom_joint_fpn[layer] = input_channels_fpn[layer]
                else:
                    if self.range_guidance_dist:
                        self.input_channels_dom_joint_fpn[layer] = input_channels_fpn[layer] + 1
                    else:
                        self.input_channels_dom_joint_fpn[layer] = input_channels_fpn[layer] + 2

            if self.joint_pma:
                self.input_channels_dom_joint_fpn[layer] = input_channels_fpn[layer] + 2
        ######### DOM CONTEXT ######

        if self.dom_context:
            dom_fc1, dom_fc2 = self.model_cfg.get('DOM_FC', [1024, 256])
        else:
            dom_fc1, dom_fc2 = self.model_cfg.get('DOM_FC', [1024, 1024])


        if self.dom_context:
            self.context_num = 1

            if not self.sep_fpn_dom_context:
                self.context_num += self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup

                if self.num_fpn_downup == 1:
                    self.context_num += 1
                #256 512

            if self.point_feat_in_voxel_dom:
                self.context_num += 2 # point context 256*2=512

            self.input_channels += self.context_num*dom_fc2

            for layer in self.fpn_layers:
                self.input_channels_fpn[layer] += self.context_num*dom_fc2
                # print('self.input_channels_fpn[layer] ini', layer, self.input_channels_fpn[layer])

        if self.range_guidance_new_conv_dom_context:
            self.context_num = 1

            if not self.sep_fpn_dom_context:
                self.context_num += self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup

                if self.num_fpn_downup == 1:
                    self.context_num += 1
                #256 512

            if self.point_feat_in_voxel_dom:
                self.context_num += 2 # point context 256*2=512

            self.input_channels += self.context_num*128

            for layer in self.fpn_layers:
                self.input_channels_fpn[layer] += self.context_num*128
        # print(" self.point_features_dim", self.point_features_dim)
        #############
        if self.point_interpolation:
            self.input_channels = self.input_channels + self.point_features_dim
            self.input_channels_dom = self.input_channels_dom + self.point_features_dim

        for layer in self.fpn_layers:
            if self.point_interpolation:
                self.input_channels_dom_fpn[layer] = self.input_channels_dom_fpn[layer] + self.point_features_dim
                self.input_channels_fpn[layer] = self.input_channels_fpn[layer] + self.point_features_dim
                # print('self.input_channels_dom_fpn[layer]', layer, self.input_channels_dom_fpn[layer]) # 512+128 = 640
                # print('self.input_channels_fpn[layer]', layer, self.input_channels_fpn[layer]) # 512+1024+128 = 1664

        self.conv_cls = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            self.input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.input_channels_det = self.input_channels

        self.rangeinv = self.model_cfg.get('RANGE_INV', False)
        self.keep_x = self.model_cfg.get('KEEP_X', False)
        self.keep_y = self.model_cfg.get('KEEP_Y', False)
        self.keep_xy = self.model_cfg.get('KEEP_XY', False)
        self.center_xy = self.model_cfg.get('CENTER_XY', False)

        self.rm_thresh = self.model_cfg.get('RM_THRESH', 0)

        if self.voxel_dom_patch_attention:

            if self.double_pma:
                if self.joint_pma:
                    self.att_patch_layer_double = LocationAttentionLayer(self.input_channels_dom_joint, self.model_cfg.PATCH_SIZE, self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE), self.no_sigmoid)
                else:
                    self.att_patch_layer_double = LocationAttentionLayer(self.input_channels_dom, self.model_cfg.PATCH_SIZE, self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE), self.no_sigmoid)

            if self.joint_two_dom:

                if self.two_attention_max:
                    self.att_patch_layer = LocationAttentionDoubleLayer(self.input_channels_dom_joint, self.model_cfg.PATCH_SIZE, self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE), self.no_sigmoid)
                else:
                    self.att_patch_layer = LocationAttentionLayer(self.input_channels_dom_joint, self.model_cfg.PATCH_SIZE, self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE), self.no_sigmoid)
            else:
                if self.two_attention_max:
                    self.att_patch_layer = LocationAttentionDoubleLayer(self.input_channels_dom, self.model_cfg.PATCH_SIZE, self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE), self.no_sigmoid)
                else:
                    self.att_patch_layer = LocationAttentionLayer(self.input_channels_dom, self.model_cfg.PATCH_SIZE, self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE), self.no_sigmoid)


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                self.input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        # print("dom_fc ", dom_fc1, dom_fc2)
        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None):

            if self.range_guidance_new_conv_dom:
                # print("input_channels_dom", input_channels_dom)
                self.conv_dom_layers = LocalDomainClassifier(input_channels=self.input_channels_dom)

                if self.sep_two_dom or (self.double_pma and not self.joint_pma):
                    self.domain_pool2 = nn.AdaptiveAvgPool2d(1)
                    self.domain_classifier2 = nn.Sequential(nn.Linear(self.input_channels_dom_sep, dom_fc1),
                                                        nn.ReLU(True), nn.Dropout(),
                                                        nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                        nn.Dropout(), nn.Linear(dom_fc2, 1))
                if self.joint_two_dom or (self.double_pma and self.joint_pma):
                    self.domain_pool2 = nn.AdaptiveAvgPool2d(1)
                    self.domain_classifier2 = nn.Sequential(nn.Linear(self.input_channels_dom_joint, dom_fc1),
                                                        nn.ReLU(True), nn.Dropout(),
                                                        nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                        nn.Dropout(), nn.Linear(dom_fc2, 1))

            else:
                self.domain_pool = nn.AdaptiveAvgPool2d(1)
                self.domain_classifier = nn.Sequential(nn.Linear(self.input_channels_dom, dom_fc1),
                                                    nn.ReLU(True), nn.Dropout(),
                                                    nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                    nn.Dropout(), nn.Linear(dom_fc2, 1))

        ######## FPN detector #########

        self.conv_cls_fpn = nn.ModuleDict()
        self.conv_box_fpn = nn.ModuleDict()

        self.att_spatial_se_layer_fpn = nn.ModuleDict()
        self.att_se_layer_fpn = nn.ModuleDict()
        self.att_patch_layer_fpn = nn.ModuleDict()


        self.att_spatial_se_layer_det_fpn = nn.ModuleDict()
        self.att_se_layer_det_fpn = nn.ModuleDict()
        self.att_patch_layer_det_fpn = nn.ModuleDict()

        if self.double_pma:
            self.att_patch_layer_fpn_double = nn.ModuleDict()
            self.att_patch_layer_det_fpn_double = nn.ModuleDict()


        # for layer in self.fpn_layers:
        #     print("self.input_channels_fpn[layer] fi", layer, self.input_channels_fpn[layer])

        for layer in self.fpn_layers:

            if self.voxel_dom_patch_attention:

                if self.double_pma:
                    if self.joint_pma:
                        self.att_patch_layer_fpn_double[layer] = LocationAttentionLayer(self.input_channels_dom_joint_fpn[layer], self.model_cfg.PATCH_SIZE_FPN[layer], self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer], self.no_sigmoid)
                    else:
                        self.att_patch_layer_fpn_double[layer] = LocationAttentionLayer(self.input_channels_dom_fpn[layer], self.model_cfg.PATCH_SIZE_FPN[layer], self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer], self.no_sigmoid)

                if self.joint_two_dom:
                    if self.two_attention_max:
                        self.att_patch_layer_fpn[layer] = LocationAttentionDoubleLayer(self.input_channels_dom_joint_fpn[layer], self.model_cfg.PATCH_SIZE_FPN[layer], self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer], self.no_sigmoid)
                    else:
                        self.att_patch_layer_fpn[layer] = LocationAttentionLayer(self.input_channels_dom_joint_fpn[layer], self.model_cfg.PATCH_SIZE_FPN[layer], self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer], self.no_sigmoid)
                else:
                    if self.two_attention_max:
                        self.att_patch_layer_fpn[layer] = LocationAttentionDoubleLayer(self.input_channels_fpn[layer], self.model_cfg.PATCH_SIZE_FPN[layer], self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer], self.no_sigmoid)
                    else:
                        self.att_patch_layer_fpn[layer] = LocationAttentionLayer(self.input_channels_fpn[layer], self.model_cfg.PATCH_SIZE_FPN[layer], self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer], self.no_sigmoid)


            self.num_anchors_per_location_fpn[layer] = sum(self.num_anchors_per_location_fpn[layer]) # 2, 7

            self.conv_cls_fpn[layer] = nn.Conv2d(
                self.input_channels_fpn[layer], self.num_anchors_per_location_fpn[layer] * self.num_class,
                kernel_size=1
            )# 512 -> 2
            self.conv_box_fpn[layer] = nn.Conv2d(
                self.input_channels_fpn[layer], self.num_anchors_per_location_fpn[layer] * self.box_coder.code_size,
                kernel_size=1
            )# 512 -> 14

        ######### fpn dir clf #########
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:

            self.conv_dir_cls_fpn = nn.ModuleDict()
            for layer in self.fpn_layers:
                self.conv_dir_cls_fpn[layer] = nn.Conv2d(
                    self.input_channels_fpn[layer],
                    self.num_anchors_per_location_fpn[layer] * self.model_cfg.NUM_DIR_BINS,
                    kernel_size=1
                )
        else:
            for layer in self.fpn_layers:
                self.conv_dir_cls_fpn[layer] = None

        # print("USE_DOMAIN_CLASSIFIER", self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None))
        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None):

            self.domain_pool_fpn = nn.ModuleDict()
            self.domain_classifier_fpn = nn.ModuleDict()
            self.conv_dom_layers_fpn = nn.ModuleDict()
            if self.sep_two_dom or self.joint_two_dom or self.double_pma:
                self.domain_pool2_fpn = nn.ModuleDict()
                self.domain_classifier2_fpn = nn.ModuleDict()

            for layer in self.fpn_layers:
                self.domain_pool_fpn[layer] = nn.AdaptiveAvgPool2d(1)
                self.domain_classifier_fpn[layer] = nn.Sequential(nn.Linear(self.input_channels_dom_fpn[layer], dom_fc1),
                                                nn.ReLU(True), nn.Dropout(),
                                                nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                nn.Dropout(), nn.Linear(dom_fc2, 1))

                if self.range_guidance_new_conv_dom:
                        # print("input_channels_dom", input_channels_dom)
                    self.conv_dom_layers_fpn[layer] = LocalDomainClassifier(input_channels=self.input_channels_dom_fpn[layer], context=self.range_guidance_new_conv_dom_context)

                    if self.sep_two_dom or (self.double_pma and not self.joint_pma):
                        self.domain_pool2_fpn[layer] = nn.AdaptiveAvgPool2d(1)
                        self.domain_classifier2_fpn[layer] = nn.Sequential(nn.Linear(self.input_channels_dom_sep_fpn[layer], dom_fc1),
                                                            nn.ReLU(True), nn.Dropout(),
                                                            nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                                nn.Dropout(), nn.Linear(dom_fc2, 1))
                        # print("sep")
                    if self.joint_two_dom or (self.double_pma and self.joint_pma):
                        self.domain_pool2_fpn[layer] = nn.AdaptiveAvgPool2d(1)
                        self.domain_classifier2_fpn[layer] = nn.Sequential(nn.Linear(self.input_channels_dom_joint_fpn[layer], dom_fc1),
                                                            nn.ReLU(True), nn.Dropout(),
                                                            nn.Linear(dom_fc1, dom_fc2), nn.ReLU(True),
                                                            nn.Dropout(), nn.Linear(dom_fc2, 1))
                        # print("joint")


        if self.range_guidance:
            if self.fov:
                total_range_x = self.model_cfg.PATCH_SIZE
                total_range_y = self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE)
                half_range_x = int(total_range_x * 0.5)
                self.x_range_matrix = torch.abs(torch.arange(0, total_range_y, 1).float()).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,1, total_range_x, 1).cuda()
                # print('x_range', x_range)
                self.y_range_matrix = torch.abs(torch.arange(-half_range_x, half_range_x, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(1,1,1,total_range_y).cuda()
                if self.range_guidance_dist:
                    joint_range_matrix = torch.stack((self.x_range_matrix,self.y_range_matrix),dim=-1).view(-1,2)
                    center_matrix = torch.tensor([(half_range_x, 0)]).float().cuda()
                    self.range_matrix = torch.cdist(joint_range_matrix,center_matrix).cuda().view(1,1,total_range_x, total_range_y)
            else:
                total_range_x = self.model_cfg.PATCH_SIZE
                total_range_y = self.model_cfg.get('PATCH_SIZE2', self.model_cfg.PATCH_SIZE)
                half_range_x = int(total_range_x * 0.5)
                half_range_y = int(total_range_y * 0.5)
                self.x_range_matrix = torch.abs(torch.arange(-half_range_y, half_range_y, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,1, total_range_x, 1).cuda()
                self.y_range_matrix = torch.abs(torch.arange(-half_range_x, half_range_x, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(1,1,1,total_range_y).cuda()
                if self.range_guidance_dist:
                    joint_range_matrix = torch.stack((self.x_range_matrix,self.y_range_matrix),dim=-1).view(-1,2)
                    center_matrix = torch.tensor([(0., 0.)]).float().cuda()
                    self.range_matrix = torch.cdist(joint_range_matrix,center_matrix).view(1,1,total_range_x, total_range_y)

            self.x_range_matrix_fpn = {}
            self.y_range_matrix_fpn = {}
            self.range_matrix_fpn = {}

            for layer in self.fpn_layers:
                if self.fov:
                    total_range_x = self.model_cfg.PATCH_SIZE_FPN[layer]
                    total_range_y = self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer]
                    half_range_x = int(total_range_x * 0.5)
                    self.x_range_matrix_fpn[layer] = torch.abs(torch.arange(0, total_range_y, 1).float()).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,1, total_range_x, 1).cuda()
                    # print('x_range', x_range)
                    self.y_range_matrix_fpn[layer] = torch.abs(torch.arange(-half_range_x, half_range_x, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(1,1,1,total_range_y).cuda()
                    if self.range_guidance_dist:
                        joint_range_matrix = torch.stack((self.x_range_matrix_fpn[layer],self.y_range_matrix_fpn[layer]),dim=-1).view(-1,2)
                        center_matrix = torch.tensor([(half_range_x, 0)]).float().cuda()
                        self.range_matrix_fpn[layer] = torch.cdist(joint_range_matrix,center_matrix).cuda().view(1,1,total_range_x, total_range_y)
                else:
                    total_range_x = self.model_cfg.PATCH_SIZE_FPN[layer]
                    total_range_y = self.model_cfg.get('PATCH_SIZE_FPN2', self.model_cfg.PATCH_SIZE_FPN)[layer]
                    half_range_x = int(total_range_x * 0.5)
                    half_range_y = int(total_range_y * 0.5)
                    self.x_range_matrix_fpn[layer] = torch.abs(torch.arange(-half_range_y, half_range_y, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1,1, total_range_x, 1).cuda()
                    self.y_range_matrix_fpn[layer] = torch.abs(torch.arange(-half_range_x, half_range_x, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(1,1,1,total_range_y).cuda()
                    if self.range_guidance_dist:
                        joint_range_matrix = torch.stack((self.x_range_matrix_fpn[layer],self.y_range_matrix_fpn[layer]),dim=-1).view(-1,2)
                        center_matrix = torch.tensor([(0, 0)]).float().cuda()
                        self.range_matrix_fpn[layer] = torch.cdist(joint_range_matrix,center_matrix).cuda().view(1,1,total_range_x, total_range_y)
        self.domain_pool = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

        for layer in self.fpn_layers:
            nn.init.constant_(self.conv_cls_fpn[layer].bias, -np.log((1 - pi) / pi))
            nn.init.normal_(self.conv_box_fpn[layer].weight, mean=0, std=0.001)

    def local_attention(self, features, d):
        # features.size() =  [1, 256, h, w]
        # d.size() = [1, 1, h, w]  after sigmoid

        d = d.clamp(1e-6, 1)
        H = - ( d * d.log() + (1-d) * (1-d).log() )
        w = 1 - H
        features_new = (1 + w) * features

        return  features_new

    def forward(self, data_dict):
        t_mode = data_dict['t_mode']
        l = data_dict['l']

        if 'pseudo' in t_mode:
            pseudo = True
        else:
            pseudo = False

        if t_mode == 'dom_img_src':
            dom_src = True
        elif t_mode == 'dom_img_tgt':
            dom_src = False
        else:
            dom_src = None

        spatial_features_2d_fpn_det = {}

        spatial_features_2d = data_dict['spatial_features_2d']
        # print("spatial_features_2d", spatial_features_2d.shape)

        range_fpn_dict = {'short': '3', 'mid': '4', 'long': '5'}
        fpn_range_dict = {'3':'short', '4':'mid', '5':'long'}

        if t_mode == 'tsne':
            return_dict = {}
            spatial_features_2d = data_dict[f'spatial_features_2d']
            return_dict[f'tsne_spatial_features_2d'] = self.domain_pool(spatial_features_2d)

            if self.voxel_dom_patch_attention and self.dom_patch_first:
                spatial_features_2d = self.att_patch_layer(spatial_features_2d)
                return_dict['tsne_spatial_features_2d_PMA_First'] = self.domain_pool(spatial_features_2d)

            if self.range_guidance and self.range_guidance_dom_only:
                total_range = spatial_features_2d.shape[-1]
                half_range = int(spatial_features_2d.shape[-1] * 0.5)
                x_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range, 1).cuda()
                y_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range).cuda()
                spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)
                return_dict['tsne_spatial_features_2d_RCD'] = self.domain_pool(spatial_features_2d)

            if self.voxel_dom_patch_attention and not self.dom_patch_first:
                spatial_features_2d = self.att_patch_layer(spatial_features_2d)
                return_dict['tsne_spatial_features_2d_PMA_Late'] = self.domain_pool(spatial_features_2d)

            for l in self.fpn_layers:
                spatial_features_2d = data_dict[f'spatial_features_2d_fpn{l}']
                return_dict[f'tsne_spatial_features_2d_fpn{l}'] = self.domain_pool(spatial_features_2d)

                if self.voxel_dom_patch_attention and self.dom_patch_first:
                    spatial_features_2d = self.att_patch_layer_fpn[l](spatial_features_2d)
                    return_dict['tsne_spatial_features_2d_PMA_First_fpn{l}'] = self.domain_pool(spatial_features_2d)

                if self.range_guidance and self.range_guidance_dom_only:
                    total_range = spatial_features_2d.shape[-1]
                    half_range = int(spatial_features_2d.shape[-1] * 0.5)
                    x_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range, 1).cuda()
                    y_range = torch.abs(torch.arange(-half_range, half_range, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range).cuda()
                    spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)
                    return_dict['tsne_spatial_features_2d_RCD_fpn{l}'] = self.domain_pool(spatial_features_2d)

                if self.voxel_dom_patch_attention and not self.dom_patch_first:
                    spatial_features_2d = self.att_patch_layer_fpn[l](spatial_features_2d)
                    return_dict['tsne_spatial_features_2d_PMA_Late_fpn{l}'] = self.domain_pool(spatial_features_2d)

            return return_dict

        ######## point feat cat ########
        if self.point_feat_in_voxel_dom:
            if self.debug: print('point_feat_in_voxel_dom')
            point_features_2d = data_dict['point_features']
            point_features_avg = torch.mean(point_features_2d, -1)
            batch_point_features = point_features_avg.view(-1, self.num_keypoints)
            x_pool_point = self.point_fc(batch_point_features)

            # if self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup > 0:
            #     for layer in self.fpn_layers:
            #         point_features_2d = data_dict['point_features']
            #         point_features_avg = torch.mean(point_features_2d, -1)
            #         batch_point_features = point_features_avg.view(-1, self.num_keypoints)
            #         x_pool_point = self.point_fc(batch_point_features)

        ###################################
        # point interpolation for general dom
        if self.point_interpolation:
            if self.debug: print('point_interpolation')
            batch_size, _, bev_x, bev_y = spatial_features_2d.shape

            if self.multi_range_interpolate:
                point_feat_dim = data_dict[f'point_features_mid'].shape[-1]
            else:
                point_feat_dim = data_dict[f'point_features'].shape[-1]

            # interpolated_bev_features_joint = torch.zeros((batch_size, point_feat_dim, bev_x, bev_y)).cuda()

            bev_feat_range = {}

            if self.multi_range_interpolate:
                for i in self.range_keys:
                    bev_feat_range[i] = self.interpolate_to_bev_features_fast(data_dict[f'point_coords_{i}'][:, 1:4], data_dict[f'point_features_{i}'], data_dict[f'spatial_features_2d_fpn{range_fpn_dict[i]}'].shape, data_dict['spatial_features_stride'])

                    data_dict[f'spatial_features_2d_fpn{range_fpn_dict[i]}'] = torch.cat((data_dict[f'spatial_features_2d_fpn{range_fpn_dict[i]}'], bev_feat_range[i]), dim=1)

                interpolated_bev_features_joint = self.interpolate_to_bev_features_fast(data_dict[f'point_coords_joint'][:, 1:4], data_dict[f'point_features_joint'], data_dict[f'spatial_features_2d'].shape, data_dict['spatial_features_stride'])

                spatial_features_2d = torch.cat((spatial_features_2d, interpolated_bev_features_joint), dim=1)

                # data_dict[f'spatial_features_2d'] = spatial_features_2d

            else:

                if self.fast_interpolation:
                    bev_feat_interpolated = self.interpolate_to_bev_features_fast(data_dict[f'point_coords'][:, 1:4], data_dict[f'point_features'], spatial_features_2d.shape, data_dict['spatial_features_stride'])
                else:
                    bev_feat_interpolated = self.interpolate_to_bev_features(data_dict[f'point_coords'][:, 1:4], data_dict[f'point_features'], spatial_features_2d.shape, data_dict['spatial_features_stride'])

                spatial_features_2d = torch.cat((spatial_features_2d, bev_feat_interpolated), dim=1)

                for layer in self.fpn_layers:
                    if self.fast_interpolation:
                        bev_feat_interpolated_fpn = self.interpolate_to_bev_features_fast(data_dict[f'point_coords'][:, 1:4], data_dict[f'point_features'], data_dict[f'spatial_features_2d_fpn{layer}'].shape, data_dict['spatial_features_stride'])
                    else:
                        bev_feat_interpolated_fpn = self.interpolate_to_bev_features(data_dict[f'point_coords'][:, 1:4], data_dict[f'point_features'], data_dict[f'spatial_features_2d_fpn{layer}'].shape, data_dict['spatial_features_stride'])

                    data_dict[f'spatial_features_2d_fpn{layer}'] = torch.cat((data_dict[f'spatial_features_2d_fpn{layer}'], bev_feat_interpolated_fpn), dim=1)
                # point interpolated for fpn

        ##### assign det feature ########
        # print('transformer bnerfore', spatial_features_2d.shape) # 1, 512, 128, 128

        # H, W = spatial_features_2d.shape[-2:]
        # B = spatial_features_2d.shape[0]
        # print('H, W ', H, W ) # 128, 128

        # pos = torch.cat([
        #     self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
        #     self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        # ], dim=-1).flatten(0, 1).unsqueeze(1).unsqueeze(0).repeat(B, 1,1,1)

        # # print("pos", pos.shape)
        # # propagate through the transformer
        # spatial_features_2d = self.transformer(pos + 0.1 * spatial_features_2d.flatten(2).permute(2, 0, 1),
        #                      self.query_pos.unsqueeze(1)).transpose(0, 1) # 100, 512

        spatial_features_2d = self.transformer(spatial_features_2d)#, self.query_pos.unsqueeze(1))
        # print('transformer after', spatial_features_2d.shape)

        spatial_features_2d_det = spatial_features_2d
        for layer in self.fpn_layers:
            spatial_features_2d_fpn_det[layer] = data_dict[f'spatial_features_2d_fpn{layer}']


        if self.model_cfg.get('USE_DOMAIN_CLASSIFIER', None):
            ############ attention ###############
            if 'dom_img' in t_mode and not self.fpn_only:

                spatial_features_2d_dom = spatial_features_2d

                if self.sep_two_dom:
                    spatial_features_2d_dom_sep = spatial_features_2d_dom

                if self.voxel_dom_patch_attention and self.dom_patch_first:

                    if self.sep_two_dom:
                        spatial_features_2d_dom_sep = self.att_patch_layer(spatial_features_2d_dom_sep)
                    else:
                        spatial_features_2d_dom = self.att_patch_layer(spatial_features_2d_dom)

                    ## dom pred 2 if joint ##
                    if self.joint_two_dom:
                        x_pool2 = self.domain_pool2(spatial_features_2d_dom).view(spatial_features_2d_dom.size(0), -1)
                        if 'reg' in t_mode:
                            x_reverse2 = grad_reverse(x_pool2, l)
                        else:
                            x_reverse2 = grad_reverse(x_pool2, l*-1)
                        dom_img_preds2 = self.domain_classifier2(x_reverse2)#.

                    if self.debug: print('voxel_dom_patch_attention first')

                ########## context ###########
                if self.range_guidance and self.range_guidance_dom_only:
                    if self.debug: print('range_guidance')

                    if self.range_guidance_dist:
                        spatial_features_2d_dom = torch.cat((spatial_features_2d_dom, self.range_matrix.repeat(spatial_features_2d_dom.shape[0],1,1,1)), dim=1)

                    else:
                        spatial_features_2d_dom = torch.cat((spatial_features_2d_dom, self.x_range_matrix.repeat(spatial_features_2d_dom.shape[0],1,1,1), self.y_range_matrix.repeat(spatial_features_2d_dom.shape[0],1,1,1)), dim=1)

                if self.voxel_dom_patch_attention and not self.dom_patch_first:
                    if self.sep_two_dom:
                        spatial_features_2d_dom_sep = self.att_patch_layer(spatial_features_2d_dom)
                    elif not self.joint_two_dom:
                        spatial_features_2d_dom = self.att_patch_layer(spatial_features_2d_dom)
                    if self.debug: print('voxel_dom_patch_attention late')

                if self.range_guidance_conv_dom or self.range_guidance_new_conv_dom:

                    if self.range_guidance_new_conv_dom_attention:
                        if self.debug: print('range_guidance_conv_dom/range_guidance_new_conv_dom _attention')
                        if 'reg' in t_mode:
                            x_reverse_dom = grad_reverse(spatial_features_2d_dom, l)
                        else:
                            x_reverse_dom = grad_reverse(spatial_features_2d_dom, l*-1)
                        if self.range_guidance_new_conv_dom_context:
                            dom_img_preds, _ = self.conv_dom_layers(x_reverse_dom)
                            #print(d_pixel)
                            # if not target:
                            _, pixel_context = self.conv_dom_layers(spatial_features_2d_dom.detach())
                            if 'dom_img_det' in t_mode:
                                data_dict['dom_head_context'] = pixel_context

                        else:
                            dom_img_preds = self.conv_dom_layers(x_reverse_dom)

                        if not self.two_attention_max:
                            if self.range_guidance_dom_only:
                                spatial_features_2d_dom = self.local_attention(spatial_features_2d_dom, dom_img_preds.detach())
                            else:
                                spatial_features_2d_dom = self.local_attention(spatial_features_2d_dom, dom_img_preds.detach())

                                spatial_features_2d_det = spatial_features_2d_dom

                    else:
                        if self.debug: print('range_guidance_conv_dom/range_guidance_new_conv_dom no_attention')
                        if 'reg' in t_mode:
                            x_reverse_dom = grad_reverse(spatial_features_2d_dom, l)
                        else:
                            x_reverse_dom = grad_reverse(spatial_features_2d_dom, l*-1)
                        if self.range_guidance_new_conv_dom_context:
                            dom_img_preds, _ = self.conv_dom_layers(x_reverse_dom)
                            #print(d_pixel)
                            # if not target:
                            _, pixel_context = self.conv_dom_layers(spatial_features_2d_dom.detach())
                            if 'dom_img_det' in t_mode:
                                data_dict['dom_head_context'] = pixel_context

                        else:
                            dom_img_preds = self.conv_dom_layers(x_reverse_dom)

                    if self.range_guidance_double_dom:
                        x_pool2 = self.domain_pool(spatial_features_2d).view(spatial_features_2d.size(0), -1)
                        if 'reg' in t_mode:
                            x_reverse2 = grad_reverse(x_pool2, l)
                        else:
                            x_reverse2 = grad_reverse(x_pool2, l*-1)
                        # print("x_reverse2", x_reverse2.shape)
                        dom_img_preds2 = self.domain_classifier(x_reverse2)#.squeeze(-1)

                    if self.sep_two_dom:
                        x_pool2 = self.domain_pool2(spatial_features_2d_dom_sep).view(spatial_features_2d_dom_sep.size(0), -1)
                        if 'reg' in t_mode:
                            x_reverse2 = grad_reverse(x_pool2, l)
                        else:
                            x_reverse2 = grad_reverse(x_pool2, l*-1)
                        # print("x_reverse2", x_reverse2.shape)
                        dom_img_preds2 = self.domain_classifier2(x_reverse2)#.squeeze(-1)

                else:
                    if self.debug: print('no range_guidance_new_conv normal dom')

                    x_pool = self.domain_pool(spatial_features_2d_dom).view(spatial_features_2d_dom.size(0), -1)
                    if self.point_feat_in_voxel_dom:
                        x_pool_joint = torch.cat((x_pool, x_pool_point),dim=-1)
                    else:
                        x_pool_joint = x_pool
                    if 'reg' in t_mode:
                        x_reverse = grad_reverse(x_pool_joint, l)
                    else:
                        x_reverse = grad_reverse(x_pool_joint, l*-1)
                    dom_head_context = self.domain_classifier[:-2](x_reverse)#.squeeze(-1)

                    if 'dom_img_det' in t_mode:
                        data_dict['dom_head_context'] = dom_head_context

                    dom_img_preds = self.domain_classifier[-2:](dom_head_context)#.squeeze(-1)


                if self.voxel_dom_patch_attention and not self.dom_patch_first:

                    if self.joint_two_dom:

                        if self.patch_unplug_context:
                            range_dim = spatial_features_2d_dom.shape[1]
                            spatial_features_2d_dom = spatial_features_2d_dom[:,:range_dim-2,:,:]

                        if self.two_attention_max:
                            local_dom_att = self.local_attention(spatial_features_2d_dom, dom_img_preds.detach())
                            spatial_features_2d_dom = self.att_patch_layer(spatial_features_2d_dom, local_dom_att)
                        else:
                            spatial_features_2d_dom = self.att_patch_layer(spatial_features_2d_dom)

                        x_pool2 = self.domain_pool2(spatial_features_2d_dom).view(spatial_features_2d_dom.size(0), -1)
                        if 'reg' in t_mode:
                            x_reverse2 = grad_reverse(x_pool2, l)
                        else:
                            x_reverse2 = grad_reverse(x_pool2, l*-1)
                        # print("x_reverse2", x_reverse2.shape)
                        dom_img_preds2 = self.domain_classifier2(x_reverse2)#.

                if self.double_pma:
                    if not self.joint_pma:
                        spatial_features_2d_double = spatial_features_2d
                    else:
                        spatial_features_2d_double = spatial_features_2d_dom

                    spatial_features_2d_double = self.att_patch_layer_double(spatial_features_2d_double)

                    x_pool2_double = self.domain_pool2(spatial_features_2d_double).view(spatial_features_2d_double.size(0), -1)
                    if 'reg' in t_mode:
                        x_reverse2_double = grad_reverse(x_pool2_double, l)
                    else:
                        x_reverse2_double = grad_reverse(x_pool2_double, l*-1)
                    dom_img_preds2 = self.domain_classifier2(x_reverse2_double)

                    # if self.patch_unplug_context:
                    #     range_dim = spatial_features_2d_dom.shape[1]
                    #     spatial_features_2d_dom = spatial_features_2d_dom[:,:range_dim-2,:,:]

                    # if self.two_attention_max:
                    #     local_dom_att = self.local_attention(spatial_features_2d_dom, dom_img_preds.detach())
                    #     spatial_features_2d_dom = self.att_patch_layer(spatial_features_2d_dom, local_dom_att)
                    # else:
                    #     spatial_features_2d_dom = self.att_patch_layer(spatial_features_2d_dom)

                # if self.dom_squeeze:
                #     dom_img_preds = dom_img_preds.squeeze(-1)
                #     if self.range_guidance_double_dom or self.sep_two_dom:
                #         dom_img_preds2 = dom_img_preds2.squeeze(-1)


                self.forward_ret_dict['dom_img_preds'] = dom_img_preds

                if self.range_guidance_double_dom or self.sep_two_dom or self.joint_two_dom or self.double_pma:
                    self.forward_ret_dict['dom_img_preds2'] = dom_img_preds2

                if self.training:
                    targets_dict_dom = self.assign_targets(
                            gt_boxes=data_dict['gt_boxes'],
                            dom_src=dom_src,
                            pseudo=pseudo
                    )
                    self.forward_ret_dict.update(targets_dict_dom)

            ####################### dom fpn #####################

            if 'dom_img' in t_mode:
                if self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup > 0:
                    # print("fpn")
                    if self.debug: print('dom img fpn')

                    for layer in self.fpn_layers:

                        spatial_features_2d_fpn = data_dict[f'spatial_features_2d_fpn{layer}'] # 642
                        spatial_features_2d_fpn_dom = spatial_features_2d_fpn# 642
                        if self.sep_two_dom:
                            spatial_features_2d_sep_fpn_dom = spatial_features_2d_fpn_dom

                        if self.voxel_dom_patch_attention and self.dom_patch_first:

                            if self.sep_two_dom:
                                spatial_features_2d_sep_fpn_dom = self.att_patch_layer_fpn[layer](spatial_features_2d_sep_fpn_dom)
                            else:
                                spatial_features_2d_fpn_dom = self.att_patch_layer_fpn[layer](spatial_features_2d_fpn_dom)

                            if self.joint_two_dom:
                                x_pool2_fpn_dom = self.domain_pool2_fpn[layer](spatial_features_2d_fpn_dom).view(spatial_features_2d_fpn_dom.size(0), -1)
                                if 'reg' in t_mode:
                                    x_reverse2_fpn_dom = grad_reverse(x_pool2_fpn_dom, l)
                                else:
                                    x_reverse2_fpn_dom = grad_reverse(x_pool2_fpn_dom, l*-1)
                                dom_img_preds2_fpn = self.domain_classifier2_fpn[layer](x_reverse2_fpn_dom)#.

                        if self.range_guidance and self.range_guidance_dom_only:
                            if self.range_guidance_dist:
                                spatial_features_2d_fpn_dom = torch.cat((spatial_features_2d_fpn_dom, self.range_matrix_fpn[layer].repeat(spatial_features_2d_fpn_dom.shape[0],1,1,1)), dim=1)
                            else:
                                spatial_features_2d_fpn_dom = torch.cat((spatial_features_2d_fpn_dom, self.x_range_matrix_fpn[layer].repeat(spatial_features_2d_fpn_dom.shape[0],1,1,1), self.y_range_matrix_fpn[layer].repeat(spatial_features_2d_dom.shape[0],1,1,1)), dim=1)

                        if self.voxel_dom_patch_attention and not self.dom_patch_first:
                            if self.sep_two_dom:
                                spatial_features_2d_sep_fpn_dom = self.att_patch_layer_fpn[layer](spatial_features_2d_sep_fpn_dom)
                            elif not self.joint_two_dom:
                                spatial_features_2d_fpn_dom = self.att_patch_layer_fpn[layer](spatial_features_2d_fpn_dom)

                        if self.range_guidance_conv_dom or self.range_guidance_new_conv_dom:
                            # x_pool = self.domain_pool().view(spatial_features_2d.size(0), -1)
                            # print('t_mode', t_mode)
                            # print("l", l)
                            if self.range_guidance_new_conv_dom_attention:
                                if 'reg' in t_mode:
                                    x_reverse_fpn_dom = grad_reverse(spatial_features_2d_fpn_dom, l)
                                else:
                                    x_reverse_fpn_dom = grad_reverse(spatial_features_2d_fpn_dom, l*-1)
                                if self.range_guidance_new_conv_dom_context:
                                    dom_img_fpn_preds, _ = self.conv_dom_layers_fpn[layer](x_reverse_fpn_dom)
                                    #print(d_pixel)
                                    # if not target:
                                    _, pixel_context_fpn = self.conv_dom_layers_fpn[layer](spatial_features_2d_fpn_dom.detach())
                                    if 'dom_img_det' in t_mode:
                                        data_dict[f'dom_head_context_fpn{layer}'] = pixel_context_fpn
                                else:
                                    dom_img_preds_fpn = self.conv_dom_layers_fpn[layer](x_reverse_fpn_dom)


                                if not self.two_attention_max:
                                    if self.range_guidance_dom_only:
                                        spatial_features_2d_fpn_dom = self.local_attention(spatial_features_2d_fpn_dom, dom_img_preds_fpn.detach())
                                    else:
                                        spatial_features_2d_fpn_dom = self.local_attention(spatial_features_2d_fpn_dom, dom_img_preds_fpn.detach())

                                        spatial_features_2d_fpn_det[layer] = spatial_features_2d_fpn_dom

                            else:
                                if 'reg' in t_mode:
                                    x_reverse_fpn_dom = grad_reverse(spatial_features_2d_fpn_dom, l)
                                else:
                                    x_reverse_fpn_dom = grad_reverse(spatial_features_2d_fpn_dom, l*-1)
                                if self.range_guidance_new_conv_dom_context:
                                    dom_img_preds_fpn, _ = self.conv_dom_layers_fpn[layer](x_reverse_fpn_dom)
                                    #print(d_pixel)
                                    # if not target:
                                    _, pixel_context_fpn = self.conv_dom_layers_fpn[layer](spatial_features_2d_fpn_dom.detach())
                                    if 'dom_img_det' in t_mode:
                                        data_dict['dom_head_context'] = pixel_context_fpn
                                else:
                                    dom_img_preds_fpn = self.conv_dom_layers_fpn[layer](x_reverse_fpn_dom)

                            if self.sep_two_dom:
                                x_pool2_sep_fpn = self.domain_pool2_fpn[layer](spatial_features_2d_sep_fpn_dom).view(spatial_features_2d_sep_fpn_dom.size(0), -1)
                                if 'reg' in t_mode:
                                    x_reverse2_sep_fpn = grad_reverse(x_pool2_sep_fpn, l)
                                else:
                                    x_reverse2_sep_fpn = grad_reverse(x_pool2_sep_fpn, l*-1)
                                # print("x_reverse2", x_reverse2.shape)
                                dom_img_preds2_fpn = self.domain_classifier2_fpn[layer](x_reverse2_sep_fpn)#.squeeze(-1)


                        else:
                            x_pool_fpn = self.domain_pool_fpn[layer](spatial_features_2d_fpn_dom).view(spatial_features_2d_fpn_dom.size(0), -1)
                            # print("x_pool_fpn", x_pool_fpn.shape)
                            if self.point_feat_in_voxel_dom:
                                x_pool_joint_fpn = torch.cat((x_pool_fpn, x_pool_point),dim=-1)
                            else:
                                x_pool_joint_fpn = x_pool_fpn
                            if 'reg' in t_mode:
                                x_reverse_fpn = grad_reverse(x_pool_joint_fpn, l)
                            else:
                                x_reverse_fpn = grad_reverse(x_pool_joint_fpn, l*-1)
                            # print("x_reverse_fpn", x_reverse_fpn.shape)
                            dom_head_context_fpn = self.domain_classifier_fpn[layer][:-2](x_reverse_fpn)#.squeeze(-1)

                            if 'dom_img_det' in t_mode:
                                data_dict[f'dom_head_context_fpn{layer}'] = dom_head_context_fpn

                            dom_img_preds_fpn = self.domain_classifier_fpn[layer][-2:](dom_head_context_fpn).squeeze(-1)

                        if self.voxel_dom_patch_attention and not self.dom_patch_first:
                            if self.joint_two_dom:
                                if self.two_attention_max:
                                    local_dom_att_fpn = self.local_attention(spatial_features_2d_fpn_dom, dom_img_preds_fpn.detach())
                                    spatial_features_2d_fpn_dom = self.att_patch_layer_fpn[layer](spatial_features_2d_fpn_dom, local_dom_att_fpn)

                                    x_pool2_fpn = self.domain_pool2_fpn[layer](spatial_features_2d_fpn_dom).view(spatial_features_2d_fpn_dom.size(0), -1)
                                    if 'reg' in t_mode:
                                        x_reverse2_fpn = grad_reverse(x_pool2_fpn, l)
                                    else:
                                        x_reverse2_fpn = grad_reverse(x_pool2_fpn, l*-1)
                                    # print("x_reverse2", x_reverse2.shape)
                                    dom_img_preds2_fpn = self.domain_classifier2_fpn[layer](x_reverse2_fpn)#.


                                else:
                                    if self.patch_unplug_context:
                                        range_dim = spatial_features_2d_fpn_dom.shape[1]
                                        spatial_features_2d_fpn_dom = spatial_features_2d_fpn_dom[:,:range_dim-2,:,:]
                                    spatial_features_2d_fpn_dom = self.att_patch_layer_fpn[layer](spatial_features_2d_fpn_dom)

                                    x_pool2_fpn = self.domain_pool2_fpn[layer](spatial_features_2d_fpn_dom).view(spatial_features_2d_fpn_dom.size(0), -1)
                                    if 'reg' in t_mode:
                                        x_reverse2_fpn = grad_reverse(x_pool2_fpn, l)
                                    else:
                                        x_reverse2_fpn = grad_reverse(x_pool2_fpn, l*-1)
                                    # print("x_reverse2", x_reverse2.shape)
                                    dom_img_preds2_fpn = self.domain_classifier2_fpn[layer](x_reverse2_fpn)#.

                        if self.double_pma:

                            if not self.joint_pma:
                                spatial_features_2d_fpn_double = spatial_features_2d_fpn
                            else:
                                spatial_features_2d_fpn_double = spatial_features_2d_fpn_dom

                            spatial_features_2d_fpn_double = self.att_patch_layer_fpn_double[layer](spatial_features_2d_fpn_double)
                            x_pool2_fpn = self.domain_pool2_fpn[layer](spatial_features_2d_fpn_double).view(spatial_features_2d_fpn_double.size(0), -1)
                            if 'reg' in t_mode:
                                x_reverse2_fpn = grad_reverse(x_pool2_fpn, l)
                            else:
                                x_reverse2_fpn = grad_reverse(x_pool2_fpn, l*-1)
                            # print("x_reverse2", x_reverse2.shape)
                            dom_img_preds2_fpn = self.domain_classifier2_fpn[layer](x_reverse2_fpn)#.

                        # if self.dom_squeeze:
                        #     dom_img_preds = dom_img_preds.squeeze(-1)
                        #     if self.range_guidance_double_dom or self.sep_two_dom:
                        #         dom_img_preds2 = dom_img_preds2.squeeze(-1)

                        self.forward_ret_dict[f'dom_img_preds_fpn{layer}'] = dom_img_preds_fpn

                        if self.sep_two_dom or self.joint_two_dom or self.double_pma:
                            self.forward_ret_dict[f'dom_img_preds2_fpn{layer}'] = dom_img_preds2_fpn

                        if self.training:
                            targets_dict_dom = self.assign_targets(
                                    gt_boxes=data_dict['gt_boxes'],
                                    dom_src=dom_src,
                                    pseudo=pseudo,
                                    fpn_layer=layer
                            )
                            self.forward_ret_dict.update(targets_dict_dom)
                    # if 'det'

        ################# det #####################
        if 'det' in t_mode:
            if not self.fpn_only:
                if self.debug: print('det img')

                if self.range_guidance and not self.range_guidance_dom_only:
                    if self.debug: print('range_guidance det')

                    total_range_x = spatial_features_2d.shape[-2]
                    total_range_y = spatial_features_2d.shape[-1]
                    half_range_x = int(spatial_features_2d.shape[-2] * 0.5)
                    half_range_y = int(spatial_features_2d.shape[-1] * 0.5)
                    x_range = torch.abs(torch.arange(-half_range_y, half_range_y, 1).float() + 0.5).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1, total_range_x, 1).cuda()
                    y_range = torch.abs(torch.arange(-half_range_x, half_range_x, 1).float() + 0.5).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(spatial_features_2d.shape[0],1,1,total_range_y).cuda()

                    spatial_features_2d = torch.cat((spatial_features_2d, x_range, y_range), dim=1)
                    # print("spatial_features_2d", spatial_features_2d.shape)

                if self.joint_attention:
                    if self.debug: print('joint_attention det fpn')
                    if self.voxel_det_seconv_attention and self.voxel_det_se_attention:
                        spatial_features_2d_out = torch.max(self.att_spatial_se_layer(spatial_features_2d), self.att_se_layer(spatial_features_2d))
                        spatial_features_2d_det = spatial_features_2d_out
                    elif self.voxel_det_seconv_attention:
                        # print("spatial_features_2d before", spatial_features_2d.shape)
                        spatial_features_2d_det = self.att_spatial_se_layer(spatial_features_2d)
                    elif self.voxel_det_se_attention:
                        spatial_features_2d_det = self.att_se_layer(spatial_features_2d)

                    else:
                        spatial_features_2d_det = spatial_features_2d
                else:
                    if self.voxel_det_seconv_attention and self.voxel_det_se_attention:
                        spatial_features_2d_out = torch.max(self.att_spatial_se_layer_det(spatial_features_2d), self.att_se_layer_det(spatial_features_2d))
                        spatial_features_2d_det = spatial_features_2d_out
                    elif self.voxel_det_seconv_attention:
                        # print("spatial_features_2d before", spatial_features_2d.shape)
                        spatial_features_2d_det = self.att_spatial_se_layer_det(spatial_features_2d)
                    elif self.voxel_det_se_attention:
                        spatial_features_2d_det = self.att_se_layer_det(spatial_features_2d)
                    else:
                        spatial_features_2d_det = spatial_features_2d

                # if self.dom_context:
                #     dom_head_context = data_dict['dom_head_context']
                #     dom_head_context_fpn = []
                #     for layer in self.fpn_layers:
                #         dom_head_context_fpn.append(data_dict[f'dom_head_context_fpn{layer}'])

                #     if self.sep_fpn_dom_context:
                #         dom_head_context_all = dom_head_context#torch.cat((dom_head_context_all, dom_head_context), dim=1)

                #     else:
                #         dom_head_context_all = torch.cat(dom_head_context_fpn, dim=1)

                #         dom_head_context_all = torch.cat((dom_head_context_all, dom_head_context), dim=1) #dom_point_context

                #     dom_head_context_all_reshape = dom_head_context_all.unsqueeze(-1).unsqueeze(-1).repeat(1,1,spatial_features_2d_det.shape[-2],spatial_features_2d_det.shape[-1])

                #     spatial_features_2d_context = torch.cat((spatial_features_2d_det, dom_head_context_all_reshape), dim=1)
                #     spatial_features_2d_det = spatial_features_2d_contextrange_guidance_new_conv_dom_context)
                if self.range_guidance_new_conv_dom_context:
                    if self.debug: print('range_guidance_new_conv_dom_context det')
                    dom_head_context = data_dict['dom_head_context']
                    # print("dom_head_context", dom_head_context.shape)
                    # print("spatial_features_2d_det", spatial_features_2d_det.shape)
                    dom_head_context_fpn = []
                    for layer in self.fpn_layers:
                        dom_head_context_fpn.append(data_dict[f'dom_head_context_fpn{layer}'])

                    if self.sep_fpn_dom_context:
                        dom_head_context_all = dom_head_context#torch.cat((dom_head_context_all, dom_head_context), dim=1)
                    else:
                        dom_head_context_all = torch.cat(dom_head_context_fpn, dim=1)

                        dom_head_context_all = torch.cat((dom_head_context_all, dom_head_context), dim=1) #dom_point_context

                    # print("dom_head_context_all", dom_head_context_all.shape)
                    #.unsqueeze(-1).unsqueeze(-1)
                    dom_head_context_all_reshape = dom_head_context_all.repeat(1,1,spatial_features_2d_det.shape[-2],spatial_features_2d_det.shape[-1])

                    spatial_features_2d_context = torch.cat((spatial_features_2d_det, dom_head_context_all_reshape), dim=1)
                    spatial_features_2d_det = spatial_features_2d_context

                cls_preds = self.conv_cls(spatial_features_2d_det)
                box_preds = self.conv_box(spatial_features_2d_det)

                cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

                self.forward_ret_dict['cls_preds'] = cls_preds
                self.forward_ret_dict['box_preds'] = box_preds

                if self.conv_dir_cls is not None:
                    dir_cls_preds = self.conv_dir_cls(spatial_features_2d_det)
                    dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                    self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
                else:
                    dir_cls_preds = None

                if self.training:
                    if pseudo:
                        pseudo_weights = data_dict['pseudo_weights']
                    else:
                        pseudo_weights = None

                    # print("gt_classes", data_dict['gt_classes'].shape)
                    # print("gt_classes", data_dict['gt_classes'])
                    # print("pseudo_weights", pseudo_weights)

                    targets_dict = self.assign_targets(
                        gt_boxes=data_dict['gt_boxes'],
                        pseudo=pseudo,
                        pseudo_weights=pseudo_weights
                    )

                    self.forward_ret_dict.update(targets_dict)

                if not self.training or self.predict_boxes_when_training:
                    batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                        batch_size=data_dict['batch_size'],
                        cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
                    )
                    data_dict['batch_cls_preds'] = batch_cls_preds
                    data_dict['batch_box_preds'] = batch_box_preds
                    data_dict['cls_preds_normalized'] = False

                if self.rangeinv:
                    # print("spatial_features_2d", spatial_features_2d.shape) #512,128,128
                    thresh = self.rm_thresh

                    start_dim = int(spatial_features_2d.shape[-1]/4.)
                    mid_dim = int(spatial_features_2d.shape[-1]/2.)
                    end_dim = start_dim+int(spatial_features_2d.shape[-1]/2.)

                    near_idx = torch.LongTensor([i for i in range(start_dim, mid_dim-thresh)]+[i for i in range(mid_dim+thresh, end_dim)])
                    far_idx = torch.LongTensor([i for i in range(start_dim)]+[i for i in range(end_dim, spatial_features_2d.shape[-1])])

                    if self.keep_x:
                        near_feat_2d = spatial_features_2d[:,:,:,near_idx]
                        far_feat_2d = spatial_features_2d[:,:,:, far_idx]
                    elif self.keep_y:
                        near_feat_2d = spatial_features_2d[:,:,near_idx,:]
                        far_feat_2d = spatial_features_2d[:,:,far_idx,:]

                    near_feat_2d_reverse = grad_reverse(near_feat_2d, l*-1)
                    range_pred_near = self.conv_range(near_feat_2d_reverse)
                    # print("near_range_pred", near_range_pred.shape)
                    far_feat_2d_reverse = grad_reverse(far_feat_2d, l*-1)
                    range_pred_far = self.conv_range(far_feat_2d_reverse)
                    # print("far_range_pred", far_range_pred.shape)

                    range_labels_near = torch.ones((range_pred_near.shape), dtype=torch.float32, device=spatial_features_2d.device)

                    range_labels_far = torch.zeros((range_pred_far.shape), dtype=torch.float32, device=spatial_features_2d.device)

                    targets_dict_range = {
                        'range_pred_near': range_pred_near,
                        'range_pred_far': range_pred_far,
                        'range_labels_near': range_labels_near,
                        'range_labels_far': range_labels_far,
                    }
                    self.forward_ret_dict.update(targets_dict_range)

            ############ FPN DET #############
            if self.num_fpn_up + self.num_fpn_down + self.num_fpn_downup > 0:
                # print("fpn")
                for layer in self.fpn_layers:

                    if self.range_guidance_new_conv_dom_context:
                        if self.debug: print('range_guidance_new_conv_dom_context det fpn', layer)
                        if self.sep_fpn_dom_context:
                            dom_head_context_all = data_dict[f'dom_head_context_fpn{layer}']

                        else:
                            dom_head_context = data_dict['dom_head_context']
                            dom_head_context_fpn = []
                            for l in self.fpn_layers:
                                dom_head_context_fpn.append(data_dict[f'dom_head_context_fpn{l}'])

                            dom_head_context_all = torch.cat(dom_head_context_fpn, dim=1)
                            dom_head_context_all = torch.cat((dom_head_context_all, dom_head_context), dim=1) #dom_point_context

                        dom_head_context_all_fpn_reshape = dom_head_context_all.repeat(1,1,spatial_features_2d_fpn_det[layer].shape[-1],spatial_features_2d_fpn_det[layer].shape[-1])

                        # combine with context
                        spatial_features_2d_fpn_context = torch.cat((spatial_features_2d_fpn_det[layer],  dom_head_context_all_fpn_reshape), dim=1)

                        spatial_features_2d_fpn_det[layer] = spatial_features_2d_fpn_context

                    if self.debug: print('det fpn', layer)
                    cls_preds = self.conv_cls_fpn[layer](spatial_features_2d_fpn_det[layer])
                    box_preds = self.conv_box_fpn[layer](spatial_features_2d_fpn_det[layer])

                    cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                    box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

                    # print("cls_preds2", cls_preds.shape) # 1, 252, 252, 2
                    # print("box_preds2", box_preds.shape) # 1, 252, 252, 14

                    self.forward_ret_dict[f'cls_preds_fpn{layer}'] = cls_preds
                    self.forward_ret_dict[f'box_preds_fpn{layer}'] = box_preds

                    if self.conv_dir_cls_fpn[layer] is not None:
                        dir_cls_preds = self.conv_dir_cls_fpn[layer](spatial_features_2d_fpn_det[layer])
                        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                        self.forward_ret_dict[f'dir_cls_preds_fpn{layer}'] = dir_cls_preds
                    else:
                        dir_cls_preds = None

                    if self.training:
                        if pseudo:
                            pseudo_weights = data_dict['pseudo_weights']
                        else:
                            pseudo_weights = None

                        targets_dict_fpn = self.assign_targets(
                            gt_boxes=data_dict['gt_boxes'],
                            pseudo=pseudo,
                            pseudo_weights=pseudo_weights,
                            fpn_layer=layer
                        )

                        self.forward_ret_dict.update(targets_dict_fpn)

                    if not self.training or self.predict_boxes_when_training:
                        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                            batch_size=data_dict['batch_size'],
                            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds,
                            fpn_layer=layer
                        )
                        data_dict[f'batch_cls_preds_fpn{layer}'] = batch_cls_preds
                        data_dict[f'batch_box_preds_fpn{layer}'] = batch_box_preds
                        data_dict[f'cls_preds_normalized_fpn{layer}'] = False

                        # print("data_dict fpn", data_dict[f'batch_cls_preds_fpn{layer}'])
                    # print("self.forward_ret_dict", self.forward_ret_dict)


        return data_dict