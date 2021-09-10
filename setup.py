import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="BatchNorm.py",
    version="0.1.0",
    url="https://github.com/David-Estevez/BatchNorm/",
    author="David Estevez",
    description='Implementation of "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".',
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),
    install_requires=requirements,
)
