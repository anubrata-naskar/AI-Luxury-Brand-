#!/usr/bin/env python
"""
Download required NLTK data for the application
"""
import nltk

def download_nltk_data():
    """Download required NLTK data"""
    nltk_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for package in nltk_packages:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package)
    
    print("All NLTK packages downloaded successfully.")

if __name__ == "__main__":
    download_nltk_data()
