from . import lemmatizer

def test_lemmas():
    lem = lemmatizer.lemmatize('ruibiștilor', msd='Ncmpoy', use_lex=False)
    assert lem[0][0] == 'ruibist'

    lem = lemmatizer.lemmatize('deschiliniri', msd='Ncfp-n', use_lex=False)
    assert lem[0][0] == 'deschilinire'

    lem = lemmatizer.lemmatize('mustangii', msd='Ncmpry', use_lex=False)
    assert lem[0][0] == 'mustang'

    lem = lemmatizer.lemmatize('racordajelor', msd='Ncfpoy', use_lex=False)
    assert lem[0][0] == 'racordaj'

    lem = lemmatizer.lemmatize('frumoaselor', msd='Afpfpoy', use_lex=False)
    assert lem[0][0] == 'frumos'

    lem = lemmatizer.lemmatize('fanțoșilor', msd='Afpmpoy', use_lex=False)
    assert lem[0][0] == 'fanțos' or lem[0][0] == 'fanțoș'

    lem = lemmatizer.lemmatize('regionale', msd='Afpfson', use_lex=False)
    assert lem[0][0] == 'regional'

    lem = lemmatizer.lemmatize('pleznise', msd='Vmil3s', use_lex=False)
    assert lem[0][0] == 'plezni'

    lem = lemmatizer.lemmatize('fugea', msd='Vmii3s', use_lex=False)
    assert lem[0][0] == 'fugi'
