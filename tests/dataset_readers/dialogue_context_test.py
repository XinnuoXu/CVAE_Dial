# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from coherence.dataset_readers import DialogueContextDatasetReader


class TestDialogueContextReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = DialogueContextDatasetReader()

        instances = ensure_list(reader.read("tests/fixtures/debug"))

        instance1 = {"context": "summertime it 's better for a mother and her cubs to hang out in the relative safety "
                                "of some tall grass staying close to mom is an inbuilt mechanism for survival".split(),
                     "response": "even so , fewer than half the cubs will make it to adulthood".split(),
                     "label": "pos"}

        instance2 = {"context": "watch out ! <u2> oh , what fun ! <u1> jon :".split(),
                     "response": "oh , that was great !".split(),
                     "label": "neg"}

        assert len(instances) == 20

        fields = instances[0].fields
        assert [t.text for t in fields["context"].tokens] == instance1["context"]
        assert [t.text for t in fields["response"].tokens] == instance1["response"]
        assert fields['label'].label == instance1['label']

        fields = instances[10].fields
        assert [t.text for t in fields["context"].tokens] == instance2["context"]
        assert [t.text for t in fields["response"].tokens] == instance2["response"]
        assert fields['label'].label == instance2['label']