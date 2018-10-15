from setuptools import setup, find_packages


setup(
    name='torchdata',
    version='0.1.0a0',
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy', 'h5py', 'Pillow', 'tqdm', 'termcolor'],
)
