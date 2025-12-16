import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        'retree.sklearn_utils',
        sources=['retree/sklearn_utils.pyx'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name='integration',
    ext_modules=cythonize(extensions, language_level=3),
    packages=['retree'],
    zip_safe=False
)
