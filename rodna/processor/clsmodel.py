import torch
from torch import nn
from . import _device


class CLSModel(nn.Module):
    """This model takes the input tensor, learns a bidirectional RNN
    MSD encoding scheme and then, a bidirectional RNN MSD classification scheme."""

    # RNN state size
    _conf_rnn_size_1 = 1024
    _conf_rnn_size_2 = 1024

    def __init__(self,
                 emb_input_vector_size: int,
                 lex_input_vector_size: int,
                 ctx_input_vector_size: int,
                 msd_encoding_vector_size: int,
                 output_msd_size: int,
                 drop_prob: float = 0.25
                 ):
        super().__init__()
        self._layer_rnn_1 = nn.LSTM(
            input_size=lex_input_vector_size + emb_input_vector_size,
            hidden_size=CLSModel._conf_rnn_size_1,
            batch_first=True,
            bidirectional=True
        )
        self._layer_linear_enc = nn.Linear(
            in_features=2 * CLSModel._conf_rnn_size_1,
            out_features=msd_encoding_vector_size
        )
        self._layer_rnn_2 = nn.LSTM(
            input_size=msd_encoding_vector_size,
            hidden_size=CLSModel._conf_rnn_size_2,
            batch_first=True,
            bidirectional=True
        )
        self._layer_linear_cls = nn.Linear(
            in_features=2 * CLSModel._conf_rnn_size_2 + ctx_input_vector_size,
            out_features=output_msd_size
        )
        self._layer_drop = nn.Dropout(p=drop_prob)
        self._sigmoid = nn.Sigmoid()
        self._layer_logsoftmax = nn.LogSoftmax(dim=2)
        self.to(device=_device)

    def forward(self, x):
        x_lex, x_emb, x_ctx = x
        b_size = x_emb.shape[0]
        h_0 = torch.zeros(
            2, b_size,
            CLSModel._conf_rnn_size_1).to(device=_device)
        c_0 = torch.zeros(2, b_size, CLSModel._conf_rnn_size_1).to(
            device=_device)

        # Concatenate along features dimension
        o_lex_emb_conc = torch.cat([x_lex, x_emb], dim=2)
        o_bd_rnn, (h_n, c_n) = self._layer_rnn_1(o_lex_emb_conc, (h_0, c_0))
        o_drop = self._layer_drop(o_bd_rnn)
        o_msd_enc = self._layer_linear_enc(o_drop)
        o_msd_enc = self._sigmoid(o_msd_enc)
        # End MSD encoding

        # MSD classification
        h_0 = torch.zeros(
            2, b_size,
            CLSModel._conf_rnn_size_2).to(device=_device)
        c_0 = torch.zeros(
            2, b_size,
            CLSModel._conf_rnn_size_2).to(device=_device)
        o_drop = self._layer_drop(o_msd_enc)
        o_bd_rnn, (h_n, c_n) = self._layer_rnn_2(o_drop, (h_0, c_0))
        o_drop = self._layer_drop(o_bd_rnn)
        o_drop = torch.cat([o_drop, x_ctx], dim=2)
        o_msd_cls = self._layer_linear_cls(o_drop)
        o_msd_cls = self._layer_logsoftmax(o_msd_cls)
        # End MSD classification

        return o_msd_enc, o_msd_cls
