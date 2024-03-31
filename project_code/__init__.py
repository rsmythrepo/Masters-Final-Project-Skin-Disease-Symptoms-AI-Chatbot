from importlib.metadata import version
import os
import sys

__version__ = version(__name__)

current_dir = os.path.dirname(__file__)
exercises_dir = os.path.join(current_dir, 'Masters-Final-Project-Skin-Disease-Symptoms-AI-Chatbot')
sys.path.append(exercises_dir)
