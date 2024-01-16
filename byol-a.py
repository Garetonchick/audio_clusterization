import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(script_dir, "byol-a/v2"))

# from byol_a2 import * 
from byol_a2 import models

models.AudioNTT2022Encoder()
