import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable

# Use Gpu name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(device)

# csv File Read
packet_data = pd.read_csv('packet.csv')

packet_data

# Delete Features (No, Time, Length)
packet_data = packet_data.drop(columns=['No.', 'Time', 'Length'])
packet_data

# Label name Change
packet_data.rename(columns={'Unnamed: 7': "Label"}, inplace=True)
packet_data

# Check Object Type Missing Value
# IF Exist      result True
# IF Empty(NAN) result False
packet_data.isna()

# IF Empty(NAN) Fill in Garbage data ("X")
packet_data = packet_data.fillna("X")

# IF Data is "NAN" an error occurs

# Check Object Type Missing Value
# IF Exist      result True
# IF Empty(NAN) result False

packet_data.isna()

import re

# re.sub Function explain Site
# https://velog.io/@kkxxh/python-re.sub
def remove_special_characters(text):
    cleaned_text = re.sub(r'[^\w\s]', '', str(text))  # \w: word characters, \s: whitespace
    return cleaned_text

# Delete Sell Special statement ("[", "]", etc....)
clear_packet = packet_data.applymap(remove_special_characters)

clear_packet

# 'object' to 'bytes'
for column in clear_packet.columns:
    if clear_packet[column].dtype == 'object':
        clear_packet[column] = clear_packet[column].astype(bytes)

print(clear_packet.dtypes)
clear_packet


import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

target_Destination = b'17230181'
target_Protocol = b'TCP'
target_Info = b'SYN'
no_target = b'SYN ACK'

# Label Function
def label_packet(row):
    if no_target in row["Info"]:
        return 0
    if row["Destination"] == target_Destination and row["Protocol"] == target_Protocol and target_Info in row["Info"]:
        return 1
    else:
        return 0

# Label Function apply
clear_packet["Label"] = clear_packet.apply(label_packet, axis=1)

# Result outout
clear_packet

# Result Data save
clear_packet.to_csv("labeled_packets.csv", index=False)

print("Packets saved to 'labeled_packets.csv'")
