from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Context]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            hidden (class specific):
               initial hidden state.

        Returns:k
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                   `[layers x batch x hidden]`
                * contexts for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None, soft=False):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        if soft:
            emb = input
        else:
            emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, cuda, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.cuda = cuda

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, self.cuda, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, self.cuda, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state, c, c_iter, context_lengths=None, soft=False):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(
            input, context, state, soft, c, c_iter, context_lengths=context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, input, context, state, soft, c, c_iter, context_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        if soft:
            emb = input
        else:
            emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1), c, c_iter,    # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

class LatentVaraibleModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, tgt_dict, \
                enc_approx, approx_mu, approx_logvar, \
                enc_true, true_mu, true_logvar, \
                glb, discriminator, max_gen_len, cuda, multigpu=False):
        self.multigpu = multigpu
        super(LatentVaraibleModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.enc_approx = enc_approx
        self.approx_mu = approx_mu
        self.approx_logvar = approx_logvar

        self.enc_true = enc_true
        self.true_mu = true_mu
        self.true_logvar = true_logvar

        self.glb_linear = glb
        self.discr = discriminator
        self.is_cuda = cuda
        self.tt = torch.cuda if cuda else torch

        self.max_gen_len = max_gen_len
        self.tgt_dict = tgt_dict

    def get_length(self, batch, blank=0):
        lengths = self.tt.LongTensor(batch.size()[1]).zero_()
        for row in batch:
            row = row.data.cpu().numpy()
            for j in range(0, len(row)):
                if row[j][0] != blank:
                    lengths[j]+=1
        return lengths

    def encode_context_response(self, tgt, src, lengths, encoder):
        hiddens, _ = encoder(src, lengths)
        # Sort tgt for encoding according to length
        tgt_lens = self.get_length(tgt)
        lens_sorted, id_sorted = tgt_lens.topk(tgt_lens.size()[0])
        tgt_sorted = torch.cat([tgt[:, id] for id in id_sorted], 1).view(tgt.size()[0], tgt.size()[1], -1)
        hiddens_sorted = []
        for layer in hiddens:
            item_sorted = []
            for item in layer:
                item_sorted.append(torch.cat([item[id] for id in id_sorted]))
            hiddens_sorted.append(torch.cat(item_sorted).view(layer.size()[0], layer.size()[1], -1))

        # Encode tgt
        hiddens_st, _ = encoder(tgt_sorted, lens_sorted, hidden=hiddens_sorted)

        # Sort back
        minus_id_sorted, id_sorted_rvs = (-1 * id_sorted).topk(tgt_lens.size()[0])
        hiddens_ret = []
        for layer in hiddens_st:
            item_sorted = []
            for item in layer:
                item_sorted.append(torch.cat([item[id] for id in id_sorted_rvs]))
            hiddens_ret.append(torch.cat(item_sorted).view(layer.size()[0], layer.size()[1], -1))

        return hiddens_ret

    def sampling(self, mu, logvar):
        samples = []
        for i in range(0, len(mu)):
            if self.is_cuda:
                eps = Variable(torch.randn(mu[i].size())).cuda()
            else:
                eps = Variable(torch.randn(mu[i].size()))
            samples.append(mu[i] + torch.exp(logvar[i]/2) * eps)
        return samples

    def get_latent_variable(self, tgt, src, lengths):
        hiddens_approx = self.encode_context_response(tgt, src, lengths, self.enc_approx)
        app_mu_dist = [self.approx_mu(hiddens_approx[i]) for i in range(0, len(hiddens_approx))]
        app_logvar_dist = [self.approx_logvar(hiddens_approx[i]) for i in range(0, len(hiddens_approx))]
        z_app = self.sampling(app_mu_dist, app_logvar_dist)

        hiddens_true, _ = self.enc_true(src, lengths)
        true_mu_dist = [self.true_mu(hiddens_true[i]) for i in range(0, len(hiddens_true))]
        true_logvar_dist = [self.true_logvar(hiddens_true[i]) for i in range(0, len(hiddens_true))]
        z_true = self.sampling(true_mu_dist, true_logvar_dist)
        return z_app, z_true, app_mu_dist, app_logvar_dist, true_mu_dist, true_logvar_dist

    def get_latent_variable_soft(self, src, tgt_soft, lengths, batch_size):
        # Encoding src
        hiddens, _ = self.enc_approx(src, lengths)
        # Get tgt soft embedding
        input_emb = [torch.mm(item, self.enc_approx.embeddings.word_lut.weight) for item in tgt_soft]
        input_emb = torch.cat(input_emb, dim=0).view(self.max_gen_len - 1, batch_size, -1)
        # Encoding tgt soft
        tgt_lengths = self.tt.LongTensor(batch_size).fill_(self.max_gen_len - 1)
        hiddens_approx, _ = self.enc_approx(input_emb, tgt_lengths, hidden=hiddens, soft=True)
        # Get z distribution
        est_mu_dist = [self.approx_mu(hiddens_approx[i]) for i in range(0, len(hiddens_approx))]
        est_logvar_dist = [self.approx_logvar(hiddens_approx[i]) for i in range(0, len(hiddens_approx))]
        return est_mu_dist, est_logvar_dist

    def get_latent_variable_test(self, src, lengths):
        hiddens, _ = self.enc_true(src, lengths)
        true_mu_dist = [self.true_mu(hiddens[i]) for i in range(0, len(hiddens))]
        true_logvar_dist = [self.true_logvar(hiddens[i]) for i in range(0, len(hiddens))]
        z_true = self.sampling(true_mu_dist, true_logvar_dist)
        return z_true

    def soft_decoder(self, rs_enc_state, batch_size, context, lengths, c, src):
        prob_list = []
        bos = self.tgt_dict.stoi[onmt.io.BOS_WORD]
        inp = Variable(self.tt.LongTensor(batch_size).fill_(bos).view(1, batch_size, -1))
        # First word
        c_list = self.discriminator.run(src, inp)
        c_iter = Variable(self.tt.FloatTensor(c_list).contiguous().view(len(c_list), -1))
        dec_out, dec_states, attn = self.decoder(inp, context, rs_enc_state, c, c_iter, context_lengths=lengths)

        for i in range(1, self.max_gen_len):
            dec_out = dec_out.squeeze(0)
            out = F.softmax(self.generator.forward(dec_out).view(batch_size, -1), dim=1)
            prob_list.append(out)
            soft_emb = torch.mm(out, self.decoder.embeddings.word_lut.weight).view(1, batch_size, -1)
            c_list_iter = self.discr.run_soft(src, prob_list)
            c_iter = c_list_iter.view(c_list_iter.size()[0], -1)
            dec_out, dec_states, attn = self.decoder(soft_emb, context, dec_states, c, c_iter, context_lengths=lengths, soft=True)
        return prob_list

    def forward(self, src, tgt, lengths, dec_state=None, only_mle = False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """

        tgt = tgt[:-1]

        # Seq2seq: Encode context
        enc_hidden, context = self.encoder(src, lengths)

        # Control Variable
        c_list = self.discr.run(src, tgt)
        c = Variable(self.tt.FloatTensor(c_list).view(len(c_list), -1))
        c_var = Variable(self.tt.FloatTensor(c_list).view(-1, len(c_list), 1))
        c_cat = torch.cat([c_var, c_var.clone()], 0)

        # Control Variable step by step
        c_iter = Variable(self.tt.FloatTensor(self.discr.run_iter(src, tgt)))

        # Latent Variable
        z_app, z_true, app_mu_dist, app_logvar_dist, true_mu_dist, true_logvar_dist = \
                self.get_latent_variable(tgt, src, lengths)

        lv_enc_hidden = tuple([torch.cat([enc_hidden[i], z_app[i], c_cat], z_app[i].dim()-1) for i in range(0, len(z_app))])

        # For attention
        context = self.glb_linear(context)

        # VAE decoding (Forward for Eq4)
        enc_state = self.decoder.init_decoder_state(src, context, lv_enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state, c, c_iter,
                                             context_lengths=lengths)

        if not only_mle:
            # Random sampling (Forward for Eq6 & Eq7)
            rs_z_true = self.sampling(true_mu_dist, true_logvar_dist)
            rs_c_prior = Variable(torch.randn(src.shape[1], 1))
            rs_c_prior = rs_c_prior.cuda() if self.is_cuda else rs_c_prior
            rs_enc_hidden = tuple([torch.cat([enc_hidden[i], rs_z_true[i], c_cat], rs_z_true[i].dim()-1) for i in range(0, len(rs_z_true))])
            rs_enc_state = self.decoder.init_decoder_state(src, context, rs_enc_hidden)
            soft_emb = self.soft_decoder(rs_enc_state, src.shape[1], context, lengths, rs_c_prior, src)

            # Forward for Eq6
            est_mu_dist, _ = self.get_latent_variable_soft(src, soft_emb, lengths, src.shape[1])
            loss_attr_z = sum([F.mse_loss(est_mu_dist[i], \
                Variable(self.tt.FloatTensor(rs_z_true[i].data.cpu().numpy()))) for i in range(0, len(rs_z_true))])

            # Forward for Eq7
            est_c = self.discr.run_soft(src, soft_emb)
            loss_attr_c = F.mse_loss(est_c, rs_c_prior)
        else:
            loss_attr_z = None
            loss_attr_c = None

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state, loss_attr_z, loss_attr_c, app_mu_dist, app_logvar_dist, true_mu_dist, true_logvar_dist



