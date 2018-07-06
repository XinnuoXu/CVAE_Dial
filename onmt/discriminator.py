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
        """Wrapper for the trained discriminator to make it work like GloVe."""

        def __init__(self, model_file, cuda):
            self.cuda = cuda
            self.dict_path = "./data/dialogue.vocab.pt"
            self.vocab = dict(torch.load(self.dict_path, "text"))

            # TODO fix this for multiple GPUs
            archive = load_archive(model_file, cuda_device=0 if cuda else -1)
            self.discr = Predictor.from_archive(archive, 'dialogue_context-predictor')

        def _ints_to_sents(self, vect, dict_type):
            return [' '.join([self.vocab[dict_type].itos[w] for w in s if w > 2])  # ignore <blank> <s> </s>
                    for s in vect.transpose()]

        def _ints_to_words(self, vect, dict_type):
            return [[self.vocab[dict_type].itos[w] if w > 2 else None for w in s]  # ignore <blank> <s> </s>
                    for s in vect.transpose()]

        def _predict(self, src_sents, tgt_sents):
            # avoid sending empty tgt_sents to the classifier or it'll crash
            batch = [{'context': ctx, 'response': resp}
                     for ctx, resp in zip(src_sents, tgt_sents)
                     if resp]
            if not batch:  # the whole batch is empty
                return [0.0] * len(tgt_sents)
            results = self.discr.predict_batch_json(batch)
            results = [r['class_probabilities'][r['all_labels'].index('pos')] for r in results]
            # cherry-pick results for valid sentences, fill the rest with 0's
            results.reverse()
            ret = []
            for resp in reversed(tgt_sents):
                if resp:
                    ret.append(results.pop())
                else:
                    ret.append(0.0)
            return ret

        def run(self, src, tgt):
            src = src.view(src.size()[0], -1).data.cpu().numpy()
            tgt = tgt.view(tgt.size()[0], -1).data.cpu().numpy()
            src_sents = self._ints_to_sents(src, 'src')
            tgt_sents = self._ints_to_sents(tgt, 'tgt')
            return self._predict(src_sents, tgt_sents)

        def run_iter(self, src, tgt):
            src = src.view(src.size()[0], -1).data.cpu().numpy()
            src_sents = self._ints_to_sents(src, 'src')
            tgt = tgt.view(tgt.size()[0], -1).data.cpu().numpy()
            tgt_words = self._ints_to_words(tgt, 'tgt')
            sim_list = []
            for i in range(0, tgt.shape[0]):
                tgt_sents = [' '.join(w for w in s[:i + 1] if w is not None) for s in tgt_words]
                sim_list.append(self._predict(src_sents, tgt_sents))
            return np.array(sim_list).transpose()

        def run_soft(self, src, tgt):
            # Src emb
            src = src.view(src.size()[0], -1).data.cpu().numpy()
            src_sents = self._ints_to_sents(src, 'src')

            # taking the argmax for tgt
            tgt = np.array([x.max(1)[1].data.cpu().numpy() for x in tgt])
            tgt_sents = self._ints_to_sents(tgt, 'tgt')
            return self._predict(src_sents, tgt_sents)
