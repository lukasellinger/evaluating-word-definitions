"""Setup script."""

import subprocess

import nltk


def install_spacy_model(model):
    """Install spacy model."""
    subprocess.run(["python", "-m", "spacy", "download", model], check=True)


if __name__ == "__main__":
    install_spacy_model("en_core_web_sm")
    install_spacy_model("en_core_web_lg")
    install_spacy_model("de_core_news_lg")
    nltk.download("punkt")
