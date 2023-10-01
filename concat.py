import pandas as pd

rock = pd.read_csv("Rock.csv")
pop = pd.read_csv("Pop.csv")
arabesk = pd.read_csv("Arabesk.csv")
türkü = pd.read_csv("Türkü.csv")

data = pd.concat([rock, pop, arabesk, türkü])
data.reset_index(drop=True, inplace=True)
data.to_csv("tracks.csv", index=False)
