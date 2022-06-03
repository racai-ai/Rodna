from . import morphology

def test_aclasses():
    ac = morphology.ambiguity_class('mașinii')
    assert 'Ncfsoy' in ac

    ac = morphology.ambiguity_class('Israelului')
    assert 'Np' in ac or 'Npmsoy' in ac

    ac = morphology.ambiguity_class('glimepiridă')
    assert 'Ncfsrn' in ac

    ac = morphology.ambiguity_class('glicozil-fosfatidilinositolului')
    assert 'Ncmsoy' in ac or 'Afpmsoy'

    ac = morphology.ambiguity_class('Postăvarul')
    # Should have been Npmsry
    assert 'Ncmsry' in ac or 'Npmsry' in ac

    ac = morphology.ambiguity_class('Luminița')
    assert 'Np' in ac or 'Npfsry' in ac

    ac = morphology.ambiguity_class('kmerilor')
    assert 'Ncmpoy' in ac

    ac = morphology.ambiguity_class('Al.I.Cuza')
    assert 'Np' in ac

    ac = morphology.ambiguity_class('fascist-totalitară')
    assert 'Afpfsrn' in ac

    ac = morphology.ambiguity_class('ieșenilor')
    # No Ncmpoy...
    assert 'Afpmpoy' in ac

    ac = morphology.ambiguity_class('dextromoramida')
    assert 'Ncfsry' in ac

    ac = morphology.ambiguity_class('cânt')
    assert 'Ncms-n' in ac and 'Vmip1s' in ac

    ac = morphology.ambiguity_class('Voyage')
    assert 'Np' in ac

    ac = morphology.ambiguity_class('emițătoare')
    assert 'Afpf--n' in ac or 'Afpfp-n' in ac or 'Afpfson' in ac

    ac = morphology.ambiguity_class('7451')
    assert 'Mc-s-d' in ac

    # Do not work...
    # ac = morphology.ambiguity_class('91/493/CEE')
    # assert 'Mc-s-b' in ac
    # ac = morphology.ambiguity_class('1.2.1')
    # assert 'Mc-s-b' in ac

    ac = morphology.ambiguity_class('DVD')
    assert 'Yn' in ac

    ac = morphology.ambiguity_class('C.T.C.')
    assert 'Yn' in ac

    ac = morphology.ambiguity_class('făcută')
    assert 'Afpfsrn' in ac and 'Vmp--sf' in ac

    ac = morphology.ambiguity_class('mergea')
    assert 'Vmii3s' in ac

    ac = morphology.ambiguity_class('permit')
    assert 'Vmip1s' in ac
