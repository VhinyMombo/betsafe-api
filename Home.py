from tkinter.messagebox import NO
from unittest import addModuleCleanup
import pandas as pd
import numpy as np
import streamlit as st
from BestBet import BestBet
import matplotlib.pyplot as plt
from gsheetsdb import connect
from PIL import Image

im = Image.open("ic3.png")
st.set_page_config(
    layout="wide",
    page_title="Welcome to SafeBet",
    page_icon=im#"🦈"
)
st.title('Bet Safe')

st.markdown('''

### The Application

This app is under development by Vhiny MOMBO, applied mathematics engineer and Data 
Scientist from [ENSIIE](https://www.ensiie.fr/) and 
[Paris Saclay University](https://www.universite-paris-saclay.fr/en). 
During his study, he learnt Quantitative finance and Risk management.
This why he implemented this solution which will help you to bet safer
by losing less while having a good expectation to win.
### The Analysis
The solution proposed for each bet use Risk Management theory and Probability
properties.

### Data

For this version 1.0, all you have to do is to follow the instructions step by step
You have to provide your odds (you can find it on the pages of your favorite boookmaker) and the 
probability for each events (if you have it, [coteur](https://www.coteur.com/) is a good provider, otherwise let the algorithm do it for you.)
 '''


)

# """ conn = connect()

# @st.cache(ttl=600)
# def run_query(query):
#     rows = conn.execute(query, headers = 1)
#     rows = rows.fetchall()
#     return rows

# conn.execute().

# sheet_url = st.secrets["public_gsheets_url"]
# rows = run_query(f'INSERT INTO "{sheet_url}" (A,B,C)\
#  VALUES (1.5,1.5,1.8,5)')
# for row in rows:
#     st.write(f"{row.A} has a : {row.B}:")
#  """