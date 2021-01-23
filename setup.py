import os
import setuptools
from datetime import datetime

__version__ = datetime.today().strftime('%Y.%m.%d').replace(".0", ".")  # Remove initial 0 from date, ex: 01 -> 1

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as file:
    long_description = file.read()

setuptools.setup(
    name="pygosolnp",
    version=__version__,
    author='Krister Sune Jakobsson',
    author_email='krister.s.jakobsson@gmail.com',
    description='This provides the GOSOLNP optimizaiton method.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/KristerSJakobsson/pygosolnp',
    license='Boost Software License',
    packages=setuptools.find_packages(),
    install_requires=["pysolnp"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
