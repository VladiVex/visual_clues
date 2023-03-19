from setuptools import setup, find_packages

setup(
    name='nebula3_visual_clues',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pymilvus==1.1.1",
        "python-arango==7.2.0"
    ], # add any additional packages that
    # needs to be installed along with your package. Eg: ''
    description='Visual Clues task of nebula3',
    version='0.3.8',
    url='https://github.com/NEBULA3PR0JECT/nebula3_visual_clues',
    author='Dima',
    author_email='dsivov@gmail.com',
    keywords=['pip', 'pypi']
)
