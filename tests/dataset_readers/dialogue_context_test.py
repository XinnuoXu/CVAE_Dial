# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from coherence.dataset_readers import DialogueContextDatasetReader


class TestDialogueContextReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = DialogueContextDatasetReader(segment_context=False, shuffle_examples=False)

        instances = ensure_list(reader.read("tests/fixtures/debug"))

        instance1 = {"context": "watch out ! <u2> oh , what fun ! <u1> jon :".split(),
                     "response": "oh , that was great !".split(),
                     "label": "neg"}

        instance2 = {"context": "can we eat now ? <u2> keep your shirt on . <u1> we 'll be in potter 's cove in "
                                "20 minutes .".split(),
                     "response": "ok , how about some pictures ?".split(),
                     "label": "neg"}

        assert len(instances) == 20

        fields = instances[10].fields
        assert [t.text for t in fields["context"].tokens] == instance1["context"]
        assert [t.text for t in fields["response"].tokens] == instance1["response"]
        assert fields['label'].label == instance1['label']

        fields = instances[18].fields
        assert [t.text for t in fields["context"].tokens] == instance2["context"]
        assert [t.text for t in fields["response"].tokens] == instance2["response"]
        assert fields['label'].label == instance2['label']

    def test_list_read_from_file(self):

        reader = DialogueContextDatasetReader(segment_context=True, shuffle_examples=False)

        instances = ensure_list(reader.read("tests/fixtures/debug"))

        instance1 = {"context": ["watch out !".split(), "oh , what fun !".split(), "jon :".split()],
                     "response": "oh , that was great !".split(),
                     "label": "neg"}

        instance2 = {"context": ["can we eat now ?".split(), "keep your shirt on .".split(),
                                 "we 'll be in potter 's cove in 20 minutes .".split()],
                     "response": "ok , how about some pictures ?".split(),
                     "label": "neg"}

        assert len(instances) == 20

        fields = instances[10].fields
        sents = [field.tokens for field in [sent for sent in fields['context'].field_list]]
        for a, b in zip(sents, instance1["context"]):
            assert [t.text for t in a] == b
        assert [t.text for t in fields["response"].tokens] == instance1["response"]
        assert fields['label'].label == instance1['label']

        fields = instances[18].fields
        sents = [field.tokens for field in [sent for sent in fields['context'].field_list]]
        for a, b in zip(sents, instance2["context"]):
            assert [t.text for t in a] == b
        assert [t.text for t in fields["response"].tokens] == instance2["response"]
        assert fields['label'].label == instance2['label']

    def test_read_from_unbalanced_file(self):

        reader = DialogueContextDatasetReader(segment_context=False, shuffle_examples=False)

        instances = ensure_list(reader.read("tests/fixtures/debug_unbalanced"))

        instance1 = {"context": "penguins mate for life . <u2> you 're not a penguin . <u1> we 're closest to apes , "
                                "actually , and they don 't mate for life .".split(),
                     "response": "take her back to the workhouse .".split(),
                     "label": "neg"}

        instance2 = {"context": "penguins mate for life . <u2> you 're not a penguin . <u1> we 're closest to apes , "
                                "actually , and they don 't mate for life .".split(),
                     "response": "what about tomorrow ?".split(),
                     "label": "neg"}

        assert len(instances) == 30

        # these two negative instances have been created automatically by parsing a single line in the input file
        fields = instances[10].fields
        assert [t.text for t in fields["context"].tokens] == instance1["context"]
        assert [t.text for t in fields["response"].tokens] == instance1["response"]
        assert fields['label'].label == instance1['label']

        fields = instances[11].fields
        assert [t.text for t in fields["context"].tokens] == instance2["context"]
        assert [t.text for t in fields["response"].tokens] == instance2["response"]
        assert fields['label'].label == instance2['label']
