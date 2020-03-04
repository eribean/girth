from setuptools import setup

if __name__ == '__main__':
    setup(
        name="girth", 
        packages=['girth'],
        version="0.1",
        license="MIT",
        description="A python package for Item Response Theory.",
        author='Ryan C. Sanchez',
        author_email='rsanchez44@gatech.edu',
        url = 'https://github.com/eribean/girth',
        download_url = 'https://github.com/eribean/girth/archive/0.1.tar.gz',
        keywords = ['IRT', 'Psychometrics', 'Item Response Theory'],
        install_requires = ['numpy', 'scipy'],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering', 
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',            
            'Programming Language :: Python :: 3.7',            
        ]
    )

