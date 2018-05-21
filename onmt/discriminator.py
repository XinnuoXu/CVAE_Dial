#coding=utf8

import torch
import numpy as np

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from coherence import *


# Monkey-patch for models trained with Torch 0.4 to run on 0.3.1
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# end monkey patch


class DiscrWrapper(object):

        def __init__(self, model_file, cuda):
            self.cuda = cuda
            self.tt = torch.cuda if cuda else torch
            self.dict_path = "./data/dialogue.vocab.pt"
            self.vocab = dict(torch.load(self.dict_path, "text"))

            archive = load_archive(model_file)
            self.discr = Predictor.from_archive(archive, 'dialogue_context-predictor')

        def _ints_to_sents(self, vect, dict_type):
            return [' '.join([self.vocab[dict_type].itos[w] for w in s if w != 0])
                    for s in vect.transpose()]

        def _ints_to_words(self, vect, dict_type):
            return [[self.vocab[dict_type].itos[w] for w in s if w != 0]
                    for s in vect.transpose()]

        def _predict(self, src_sents, tgt_sents):
            results = [self.discr.predict_json({'context': ctx, 'response': resp})
                       for ctx, resp in zip(src_sents, tgt_sents)]
            return [r['class_probabilities'][r['all_labels'].index('pos')] for r in results]

        def run(self, src, tgt):
            src = src.view(src.size()[0], -1).data.numpy()
            tgt = tgt.view(tgt.size()[0], -1).data.numpy()
            src_sents = self._ints_to_sents(src, 'src')
            tgt_sents = self._ints_to_sents(tgt, 'tgt')
            return self._predict(src_sents, tgt_sents)

        def run_iter(self, src, tgt):
            src = src.view(src.size()[0], -1).data.numpy()
            src_sents = self._ints_to_sents(src, 'src')
            tgt = tgt.view(tgt.size()[0], -1).data.numpy()
            tgt_words = self._ints_to_words(tgt, 'tgt')
            sim_list = []
            for i in range(0, tgt.shape[0]):
                tgt_sents = [' '.join(w for w in s[:i + 1]) for s in tgt_words]
                sim_list.append(self._predict(src_sents, tgt_sents))
            return np.array(sim_list).transpose()

        def run_soft(self, src, tgt):
            import pudb; pu.db
            # Src emb
            src = src.view(src.size()[0], -1).data.numpy()
            src_sents = self._ints_to_sents(src, 'src')

            # taking the argmax for tgt
            tgt = np.array([x.max(1)[1].data.numpy() for x in tgt])
            tgt_sents = self._ints_to_sents(tgt, 'tgt')
            return self._predict(src_sents, tgt_sents)
