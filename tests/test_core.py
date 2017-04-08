import bb_stitcher.core as core


def test_get_default_config():
    config = core.get_default_config()

    config_set = {
        'Rectificator',
        'FeatureBasedStitcher',
        'SURF',
        'FeatureMatcher'}
    assert set(config.sections()) == config_set


def test_get_default_debug_config():
    deb_config = core.get_default_debug_config()
    assert 'loggers' in deb_config
