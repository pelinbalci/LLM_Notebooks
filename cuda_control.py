import re
import numpy as np
import pandas as pd
import editdistance
import torch


# device
device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)