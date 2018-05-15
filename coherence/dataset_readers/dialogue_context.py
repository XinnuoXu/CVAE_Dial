import logging
from typing import Dict

import tqdm
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
import random

from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("dialogue_context")
class DialogueContextDatasetReader(DatasetReader):

    def __init__(self, lazy:
                 bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        examples = self._read_file(file_path + ".pos", "pos") + self._read_file(file_path + ".neg", "neg")
        random.shuffle(examples)
        for ex in examples:
            yield ex

    def _read_file(self, file_path, label):
        examples = []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from file: %s", file_path, label)
            for line_num, line in enumerate(tqdm.tqdm(data_file.readlines())):
                line = line.strip("\n")
                if not line:
                    continue
                context, response = line.split("\t")
                examples.append(self.text_to_instance(context, response, label))
        return examples

    @overrides
    def text_to_instance(self, context: str, response: str, label=None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_response = self._tokenizer.tokenize(response)
        context_field = TextField(tokenized_context, self._token_indexers)
        response_field = TextField(tokenized_response, self._token_indexers)
        fields = {'context': context_field, 'response': response_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'DialogueContextDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers)
