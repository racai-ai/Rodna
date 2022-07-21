from . import lemmatizer

def test_lemmas():
    lem = lemmatizer.lemmatize('ruibiștilor', msd='Ncmpoy', use_lex=False)
    assert lem[0][0] == 'ruibist'

    lem = lemmatizer.lemmatize('mungiii', msd='Ncmpry', use_lex=False)
    assert lem[0][0] == 'mungiu'

    lem = lemmatizer.lemmatize('deschiliniri', msd='Ncfp-n', use_lex=False)
    assert lem[0][0] == 'deschilinire'

    lem = lemmatizer.lemmatize('mustangii', msd='Ncmpry', use_lex=False)
    assert lem[0][0] == 'mustang'

    lem = lemmatizer.lemmatize('plicurilor', msd='Ncfpoy', use_lex=False)
    assert lem[0][0] == 'plic'

    lem = lemmatizer.lemmatize('frumoaselor', msd='Afpfpoy', use_lex=False)
    assert lem[0][0] == 'frumos'

    lem = lemmatizer.lemmatize('fanțoșilor', msd='Afpmpoy', use_lex=False)
    assert lem[0][0] == 'fanțos'

    lem = lemmatizer.lemmatize('regionale', msd='Afpfson', use_lex=False)
    assert lem[0][0] == 'regional'

    lem = lemmatizer.lemmatize('mersese', msd='Vmil3s', use_lex=False)
    assert lem[0][0] == 'merge'

    lem = lemmatizer.lemmatize('petrecea', msd='Vmii3s', use_lex=False)
    assert lem[0][0] == 'petrece'

    lem = lemmatizer.lemmatize('fachez', msd='Vmip1s', use_lex=False)
    assert lem[0][0] == 'faca'
