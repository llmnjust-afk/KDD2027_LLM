from setuptools import setup, find_packages

setup(
    name="kg-scaffold",
    version="0.1.0",
    description="KG-Symbolic Co-Refinement for Literature-Based Discovery (KDD 2027)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.2",
        "numpy>=1.26",
        "pandas>=2.1",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "tenacity>=8.2",
    ],
    extras_require={
        "full": [
            "pykeen>=1.10",
            "torch>=2.1",
            "sentence-transformers>=2.5",
            "openai>=1.30",
            "pyvis>=0.3",
            "streamlit>=1.33",
            "matplotlib>=3.8",
            "seaborn>=0.13",
        ],
    },
)
