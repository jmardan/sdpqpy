"""
sdpqpy: Solve quantum ground state problems with semi-definte programming.
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
setup(
    name='sdpqpy',
    version='0.1.0',
    author='Christian Gogolin',
    author_email='sdpqpy@cgogolin.de',
    packages=['sdpqpy'],
    url='https://github.com/cgogolin/sdpqpy',
    keywords=[
        'sdp',
        'semidefinite programming',
        'relaxation',
        'quantum',
        'ground state'
    ],
    license='LICENSE',
    description='Semi-definite programming for approximately solving quantum ground state problems',
    long_description=open('README.rst').read(),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python'
    ],
#    install_requires=[
#        "sympy >= 0.7.2",
#        "numpy"
#    ],
#    test_suite="tests"
)
