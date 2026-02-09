from . import splitter
from . import tagger
from . import parser

def test_parsing():
    sentences = splitter.sentence_split("Acesta este un text simplu. Urmează, bine-înțeles, o a doua propoziție.")

    tagged_sentence_1 = tagger.tag_sentence(sentences[0])
    tagged_sentence_2 = tagger.tag_sentence(sentences[1])

    parsed_sentence_1 = parser.parse_sentence(sentence=tagged_sentence_1)

    assert parsed_sentence_1[0][2] == 4 and parsed_sentence_1[0][3] == 'nsubj'
    assert parsed_sentence_1[1][2] == 4 and parsed_sentence_1[1][3] == 'cop'
    assert parsed_sentence_1[4][2] == 4 and parsed_sentence_1[4][3] == 'amod'

    parsed_sentence_2 = parser.parse_sentence(sentence=tagged_sentence_2)

    assert parsed_sentence_2[0][2] == 0 and parsed_sentence_2[0][3] == 'root'
    assert parsed_sentence_2[2][2] == 1 and parsed_sentence_2[2][3] == 'advmod'
    assert parsed_sentence_2[4][2] == 8 and parsed_sentence_2[4][3] == 'det'
    assert parsed_sentence_2[7][2] == 1 and parsed_sentence_2[7][3] == 'nsubj'
