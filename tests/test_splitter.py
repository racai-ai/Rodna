from . import splitter

def test_simple():
    sentences = splitter.sentence_split("Acesta este un text simplu. Urmează, bine-înțeles, o a doua propoziție. "
                                     "Iar textul se termină cu ultima propoziție.")
    assert len(sentences) == 3
    assert sentences[0][0][0] == 'Acesta'
    assert sentences[1][0][0] == 'Urmează'
    assert sentences[2][0][0] == 'Iar'


def test_dialog():
    sentences = splitter.sentence_split(
        "- Ce faci nene, nu te duci?! - îl întrebă Maria pe Ion.")
    assert len(sentences) == 2
    assert sentences[0][2][0] == 'Ce'
    assert sentences[1][2][0] == 'îl'


def test_enumeration():
    sentences = splitter.sentence_split("Articolul 10\n(1) Prin prezenta ne adresăm vouă, "
                                        "celor care ne întrebați.\n(2) Prezenta orientare intră sub incidența legii.")
    assert len(sentences) == 3
    assert sentences[1][1][0] == '1'
    assert sentences[2][1][0] == '2'
