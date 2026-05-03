from setuptools import setup, find_packages

setup(
    name='penguin_trainer',
    version='0.1',
    packages=find_packages(), # This will automatically find the 'trainer' directory
    install_requires=[
        'pandas',
        'xgboost',
        'scikit-learn',
        'google-cloud-bigquery',
        'google-cloud-aiplatform',
        'google-cloud-storage',
        'seaborn',
        'db-dtypes',
    ],
)