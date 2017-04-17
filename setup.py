"""This is the setup file to install the bb_stitcher."""
from pip.req import parse_requirements
from setuptools import setup


install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

setup(
    name='bb_stitcher',
    version='0.0.0.dev3',
    description='Stitch images from different cam positions,'
                'with an affine transformation',
    long_description='',
    entry_points={
            'console_scripts': [
                'bb_stitcher = bb_stitcher.scripts.bb_stitcher:main'
            ]
    },
    url='https://github.com/gitmirgut/bb_stitcher',
    install_requires=reqs,
    dependency_links=dep_links,
    author='gitmirgut',
    author_email="gitmirgut@users.noreply.github.com",
    packages=['bb_stitcher', 'bb_stitcher.picking', 'bb_stitcher.scripts'],
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5'
    ],
    package_data={
        'bb_stitcher': ['*.ini']
    }
)
