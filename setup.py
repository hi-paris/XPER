"""The setup script."""

from setuptools import setup, find_packages

with open('History.rst') as history_file:
    history = history_file.read()

requirements = []
test_requirements = []

setup(
    author="""Sebastien Saurin, Christophe Hurlin, Christophe Perignon,
     supported by Awais Sani and Gaëtan Brison""",
    author_email='engineer.hi.paris@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="""XPER (eXplainable PERformance) is a methodology designed 
    to measure the specific contribution of the input features to 
    the predictive performance of any econometric or machine learning model.""",
    install_requires=["numpy", "pandas", "scipy", "scikit-learn", "shap", "seaborn", "matplotlib", "xgboost", "statsmodels","tqdm","plotly"],
    license="MIT license",
    keywords='XPER',
    name='XPER',
    packages=find_packages(include=['XPER', 'XPER.*']),
    include_package_data=True,
    package_data = {
        '': ['*.csv'],
        'XPER': ['datasets/loan/Loan_Status.csv'],
    },
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/hi-paris/XPER',
    version='0.0.73',
    zip_safe=False,
)
