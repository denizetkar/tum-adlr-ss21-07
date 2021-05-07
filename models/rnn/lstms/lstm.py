"""
Implementation of LSTM variants.

For now, they only support a sequence size of 1, and are ideal for RL use-cases.
Besides that, they are a stripped-down version of PyTorch's RNN layers.
(no bidirectional, no num_layers, no batch_first)
"""
from typing import List, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf

    Special args:

    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True, dropout: float = 0.0, dropout_method: str = "pytorch"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        self.dropout_method = dropout_method.lower()
        assert self.dropout_method in ["pytorch", "gal", "moon", "semeniuta"]

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = Variable(th.bernoulli(th.empty(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self, gain: float = 1.0):
        for name, w in self.named_parameters():
            if "weight" in name:
                if "h2h" in name:
                    nn.init.orthogonal_(w, gain=gain)
                else:
                    nn.init.xavier_normal_(w, gain=gain)
            elif "bias" in name:
                nn.init.constant_(w, val=0.0)

    def forward(self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Calculate outputs and the final hidden state of the input sequence.

        :param x: Input sequence of shape (seq_len, batch_size, hidden_size)
        :param hidden: LSTM hidden state where h and c are of shape (batch_size, hidden_size)
        :return: Outputs and the final hidden state
        """
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        seq_len = x.size(0)
        outputs: List[th.Tensor] = []

        for i in range(seq_len):
            o, (h, c) = self._forward_one_step(x[i], (h, c), do_dropout)
            outputs.append(o)

        return th.stack(o, dim=0), (h, c)

    def _forward_one_step(
        self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor], do_dropout: bool
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        h, c = hidden
        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, : 3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size :].tanh()
        i_t = gates[:, : self.hidden_size]
        f_t = gates[:, self.hidden_size : 2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size :]

        # cell computations
        if do_dropout and self.dropout_method == "semeniuta":
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == "moon":
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == "pytorch":
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == "gal":
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        return h_t, (h_t, c_t)


class GalLSTM(LSTM):

    """
    Implementation of Gal & Ghahramami:
    'A Theoretically Grounded Application of Dropout in Recurrent Neural Networks'
    http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_method = "gal"
        self.sample_mask()


class MoonLSTM(LSTM):

    """
    Implementation of Moon & al.:
    'RNNDrop: A Novel Dropout for RNNs in ASR'
    https://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_method = "moon"
        self.sample_mask()


class SemeniutaLSTM(LSTM):
    """
    Implementation of Semeniuta & al.:
    'Recurrent Dropout without Memory Loss'
    https://arxiv.org/pdf/1603.05118.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_method = "semeniuta"


class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf

    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_method: str = "pytorch",
        ln_preact: bool = True,
        learnable: bool = True,
    ):
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, bias=bias, dropout=dropout, dropout_method=dropout_method
        )
        if ln_preact:
            self.ln_i2h = nn.LayerNorm(4 * hidden_size, elementwise_affine=learnable)
            self.ln_h2h = nn.LayerNorm(4 * hidden_size, elementwise_affine=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = nn.LayerNorm(hidden_size, elementwise_affine=learnable)

    def forward(self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Calculate outputs and the final hidden state of the input sequence.

        :param x: Input sequence of shape (seq_len, batch_size, hidden_size)
        :param hidden: LSTM hidden state where h and c are of shape (batch_size, hidden_size)
        :return: Outputs and the final hidden state
        """
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        seq_len = x.size(0)
        outputs: List[th.Tensor] = []

        for i in range(seq_len):
            o, (h, c) = self._forward_one_step(x[i], (h, c), do_dropout)
            outputs.append(o)

        return th.stack(outputs, dim=0), (h, c)

    def _forward_one_step(
        self, x: th.Tensor, hidden: Tuple[th.Tensor, th.Tensor], do_dropout: bool
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        h, c = hidden
        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = preact[:, : 3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size :].tanh()
        i_t = gates[:, : self.hidden_size]
        f_t = gates[:, self.hidden_size : 2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size :]

        # cell computations
        if do_dropout and self.dropout_method == "semeniuta":
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == "moon":
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        c_t = self.ln_cell(c_t)
        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == "pytorch":
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == "gal":
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        return h_t, (h_t, c_t)


class LayerNormGalLSTM(LayerNormLSTM):

    """
    Mixes GalLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_method = "gal"
        self.sample_mask()


class LayerNormMoonLSTM(LayerNormLSTM):

    """
    Mixes MoonLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_method = "moon"
        self.sample_mask()


class LayerNormSemeniutaLSTM(LayerNormLSTM):

    """
    Mixes SemeniutaLSTM's Dropout with Layer Normalization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_method = "semeniuta"
