"""

"""
from setuptools import setup

setup(
    name='bb_stitcher',
    version='0.0.0.dev2',
    description='Stitch images from different cam positions,'
                'with an affine transformation',
    long_description='',
    entry_points={
            'console_scripts': [
                'bb_stitcher = bb_stitcher.scripts.bb_stitcher:main'
            ]
    },
    url='https://github.com/gitmirgut/bb_stitcher',
    author='gitmirgut',
    author_email="gitmirgut@users.noreply.github.com",
    packages=['bb_stitcher', 'bb_stitcher.picking'],
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5'
    ],
    package_data={
        'bb_stitcher': ['default_config.ini']
    }
)
