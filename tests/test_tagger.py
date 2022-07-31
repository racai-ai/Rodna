from . import splitter
from . import tagger


def test_simple():
    sentences = splitter.sentence_split("Acesta este un text simplu. Urmează, bine-înțeles, o a doua propoziție. "
                                        "Iar textul se termină cu ultima propoziție.")
    assert len(sentences) == 3

    tagged_sentence_1 = tagger.tag_sentence(sentences[0])
    assert tagged_sentence_1[0][1] == 'Pd3msr'
    assert tagged_sentence_1[1][1] == 'Vaip3s'

    tagged_sentence_2 = tagger.tag_sentence(sentences[1])
    assert tagged_sentence_2[0][1] == 'Vmip3'
    assert tagged_sentence_2[7][1] == 'Ncfsrn'

    tagged_sentence_3 = tagger.tag_sentence(sentences[2])
    assert tagged_sentence_3[0][1] == 'Rc'
    assert tagged_sentence_3[5][1] == 'Mofsrly'
