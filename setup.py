import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='BesselConv',
    version='0.0.1',
    author='Valentin Delchevalerie',
    author_email='valentin.delchevalerie@unamur.be',
    description='A package that implements Bessel Convolutions',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ValDelch/B_CNNs/tree/torch-package',
    license='MIT',
    packages=['BesselConv'],
    install_requires=['numpy', 'torch', 'einops', 'scipy'],
)