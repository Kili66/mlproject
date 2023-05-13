from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    
    '''
    
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = [line.strip() for line in file_obj if not line.startswith('-e .')]
    return requirements
#Required parameters
setup(
name='mlproject',
version='0.01',
author='Mariam',
author_email='mkb9999protnmail.com',
packages= find_packages(),
install_requires=get_requirements('requirements.txt')

)