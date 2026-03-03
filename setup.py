from setuptools import setup, find_packages

setup(
    name="war_economic_impact",
    version="1.0.0",
    description=(
        "ML pipeline to predict the economic impact of armed conflicts "
        "using pre-war indicators and conflict characteristics."
    ),
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/war-economic-impact",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.3.0",
        "optuna>=3.6.0",
        "mlflow>=2.11.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "joblib>=1.3.0",
        "loguru>=0.7.0",
        "shap>=0.45.0",
    ],
    entry_points={
        "console_scripts": [
            "train=models.train_model:main",
            "predict=models.predict_model:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
