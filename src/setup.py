from setuptools import setup, find_packages

setup(
    name='yaduha',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'python-dotenv',
        'openai',
        'spacy',
        'tqdm',
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'sentence-transformers',
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cpu'  # The custom index URL for the CPU wheels
    ],
)