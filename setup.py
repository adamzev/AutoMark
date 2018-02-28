"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rd'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='AutoMark',

    # Semantic Versioning 2.0.0 https://semver.org/
    version='0.1.0', 

    description='Automaticly grade handwritten student responses on  \"Grade-it\" math worksheets.',

    # long_description=long_description,

    url='https://github.com/tutordelphia/AutoMark',

    author='Adam Levin',

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='AdamLevin@tutordelphia.com',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='computer-vision digit-recognition teaching-tool worksheets grading',  # Optional


    packages=find_packages(),

    install_requires=['numpy>=1', 'hdf5', 'pillow', 'Keras>=2', 'opencv-python>=3.4', 'pyzbar'],

    package_data={  # Optional
        'data': ['cnn_mnist.h5'],
    },
)