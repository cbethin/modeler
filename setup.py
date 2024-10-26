from setuptools import setup, find_packages

setup(
    name='modeler',
    version='0.1.6',
    description='A library for fine-tuning and serving models like Flan-T5, including integration with OpenAI GPT models.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/flan_t5_fine_tuner',  # replace with your GitHub repository URL
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',  
        'transformers>=4.9.0',
        'datasets>=1.6.0',
        'pandas>=1.1.0',
        'scikit-learn>=0.24.0',
        'psutil>=5.8.0',
        'Flask>=2.0.0',
        'openai>=0.10.2',
        'ipywidgets>=7.0.0'
    ],
    entry_points={
        'console_scripts': [
            'flan_t5_fine_tuner=flan_t5_fine_tuner.chat_server:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
