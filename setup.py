from setuptools import setup, find_packages

setup(
    name='ExpressLane', 
    version='0.1.0',
    author='Maximilian Hingerl',
    author_email='maximilian.hignerl@gmail.com',
    description='A short description of my package',
    packages=find_packages(), 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)