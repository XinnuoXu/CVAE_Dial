from typing import Tuple, List

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from overrides import overrides
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('dialogue_context-predictor')
class DialogueContextClassifierPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        context = json_dict['context']
        response = json_dict['response']
        instance = self._dataset_reader.text_to_instance(context=context,
                                                         response=response)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Tuple[Instance, JsonDict]]:
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        insts = []
        for json_dict in json_dicts:
            insts.append((self._dataset_reader.text_to_instance(json_dict['context'], json_dict['response']),
                          {'all_labels': all_labels}))
        return insts
