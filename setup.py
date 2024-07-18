from setuptools import setup, find_packages

setup(
    name='da_vis',
    version='0.1',
    description='A package for visualizing data and machine learning model performance',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sabyasachi',
    author_email='sabychakraborty08@gmail.com',
    url='https://github.com/mixter3011/da_vis.git',  # Replace with your actual GitHub repo URL
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'shap',
        'pandas',
        'dash',
        'plotly'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
