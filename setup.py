from setuptools import setup

setup(
    name='pyplates',
    version='0.1.0',    
    description='Python templates for building data science pipelines',
    url='https://github.com/YounesZ/pyplates',
    author='Younes Zerouali',
    author_email='younes_zerouali@hotmail.com',
    license='BSD 2-clause',
    packages=['pyplates'],
    install_requires=['pandas',
                      'numpy'],

    classifiers=[
        'Development Status :: 1 - Initializations launch',
        'Intended Audience :: Private Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: >=3.8',
    ],
)