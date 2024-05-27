import pandas as pd
import os
import matplotlib.pyplot as plt

confusion_matrix_path = r'csv\confusion_matrix.csv'

df = pd.read_csv(confusion_matrix_path, index_col=0, header=None)