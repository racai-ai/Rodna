from . import tokenizer

def test_simple():
    tokens = tokenizer.tokenize("Aceasta este o propoziție simplă.")
    assert len(tokens) == 10

def test_clitics():
    tokens = tokenizer.tokenize("Pe Mirciulică l-am prins cu mâța-n sac și se-ncearcă mușamalizarea cazului.")
    assert len(tokens) == 25
    assert tokens[4][0] == 'l-'
    assert tokens[12][0] == '-n'
    assert tokens[19][0] == '-ncearcă'

def test_mwes():
    tokens = tokenizer.tokenize("Plecând de la definițiile anterioare considerăm că baza de date este o colecție de date referitoare la un "
                                "domeniu de activitate particular, memorată pe un suport adresabil de date împreună cu descrierea "
                                "structurii datelor și a relațiilor dintre ele, accesibilă mai multor utilizatori.")
    assert len(tokens) == 88
    assert tokens[2][1] == 'MWE'
    assert tokens[4][1] == 'MWE'
    assert tokens[59][1] == 'MWE'
    assert tokens[61][1] == 'MWE'

def test_abbrs():
    tokens = tokenizer.tokenize("(Visual Cobol, Turbo Pascal, Visual Basic, C, C++ etc.).")
    assert len(tokens) == 25
    assert tokens[22][1] == 'ABBR'
    assert tokens[23][1] == 'ABBR'

def test_long_abbr_bug():
    tokens = tokenizer.tokenize(
        "Comitetul O.N.U. împotriva torturii cere explicații " + \
        "SUA și Marii Britanii în legătură cu tratamentele inumane " + \
        "aplicate deținuților irakieni.")
    assert len(tokens) == 41
    assert tokens[2][1] == 'ABBR'
    assert tokens[3][1] == 'ABBR'
    assert tokens[4][1] == 'ABBR'
    assert tokens[5][1] == 'ABBR'
    assert tokens[6][1] == 'ABBR'
    assert tokens[7][1] == 'ABBR'

def test_wrong_word_order_bug():
    tokens = tokenizer.tokenize(
        "Poliția timișeană a efectuat cercetări, comunicându-ne că Mădălina Constanța B., " + \
        "de 20 de ani, care conducea un autoturism Škoda, nu a adaptat viteza pe o porțiune de drum cu denivelări."
    )
    tokens = tokenizer.glue_tokens(tokens)
    assert tokens[20][0] == 'B.'
    assert tokens[21][0] == ','

def test_glued_abbr_bug():
    tokens = tokenizer.tokenize(
        "În ianuarie 1879, o cronică dramatică nesemnată ataca spectacolul " + \
        "de la Teatrul Național cu O noapte furtunoasă de I. L. Caragiale."
    )
    tokens = tokenizer.glue_tokens(tokens)
    assert tokens[-4][0] == 'L.'
    assert tokens[-6][0] == 'I.'
