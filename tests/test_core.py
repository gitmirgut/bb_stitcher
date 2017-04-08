import bb_stitcher.core as core


def test_auto_config():
    config = core.get_default_config()

    config_set = {
        'Rectificator',
        'FeatureBasedStitcher',
        'SURF',
        'FeatureMatcher'}
    assert set(config.sections()) == config_set
