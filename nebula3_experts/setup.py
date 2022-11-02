from setuptools import setup, find_packages

setup(
    name='nebula3_experts',
    packages=find_packages(),
    install_requires=[
        "nebula3_database==0.2.2",
        "cachetools",
        "opencv-python",
        "opencv-contrib-python"
    ], # add any additional packages that
    # needs to be installed along with your package. Eg: ''
    description='Base expert for nebula3',
    version='1.2.1',
    url='https://github.com/NEBULA3PR0JECT/nebula3_experts',
    author='Amir',
    author_email='aharon.amir@gmail.com',
    keywords=['pip', 'pypi', 'microservice']
)