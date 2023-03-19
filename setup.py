from setuptools import setup, find_packages
import glob
setup(
    name='nebula3_visual_clues',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[], # add any additional packages that
    # needs to be installed along with your package. Eg: ''
    package_data = {'package_fiddler': ['visual_clues/configs/*', 'visual_clues/models/*', 'visual_clues/visual_token_ontology/vg/*']},
    description='Visual Clues implementation for Nebula3',
    version='0.3.12',
    url='https://github.com/NEBULA3PR0JECT/visual_clues',
    author='Ilan',
    author_email='bebetteryou2015@gmail.com',
    keywords=['pip', 'pypi']
)
