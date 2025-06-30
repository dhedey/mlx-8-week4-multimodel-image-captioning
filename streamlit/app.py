import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
import random
import io
import sys
import os
import time
from uuid import uuid4
import streamlit.components.v1 as components

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.common import select_device

device = select_device()

# TODO!