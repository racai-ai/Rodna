from . import lemmatizer

def test_lemmas():
    lem = lemmatizer.lemmatize('canioanelor', msd='Ncfpoy', use_lex=False)
    assert lem[0][0] != 'canion'

    lem = lemmatizer.lemmatize('canioanelor', msd='Ncfpoy', use_lex=True)
    assert lem[0][0] == 'canion'

    lem = lemmatizer.lemmatize('ruibiștilor', msd='Ncmpoy', use_lex=False)
    assert lem[0][0] == 'ruibist'

    lem = lemmatizer.lemmatize('mungiii', msd='Ncmpry', use_lex=False)
    assert lem[0][0] == 'mungiu'

    lem = lemmatizer.lemmatize('descifratoarelor', msd='Ncfpoy', use_lex=False)
    assert lem[0][0] == 'descifrator'

    lem = lemmatizer.lemmatize('deschiliniri', msd='Ncfp-n', use_lex=False)
    assert lem[0][0] == 'deschilinire'

    lem = lemmatizer.lemmatize('mustangii', msd='Ncmpry', use_lex=False)
    assert lem[0][0] == 'mustang'
