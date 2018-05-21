# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase

from coherence.dataset_readers.dialogue_context import DialogueContextDatasetReader


class DialogueContextCoherenceClassifierTest(ModelTestCase):
    def setUp(self):
        super(DialogueContextCoherenceClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/dialogue_context_hierarchical_coherence_attention_classifier.json',
                          'tests/fixtures/debug')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)