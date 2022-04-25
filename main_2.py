import pandas as pd
import  numpy as np

df1 = pd.read_csv(r"Pliki\Fake.csv")
df1["Fake/True"] = [0] * (np.shape(df1)[0])

df2 = pd.read_csv(r"Pliki\True.csv")
df2["Fake/True"] = [1] * (np.shape(df2)[0])

# print(np.shape(df1))
# print(np.shape(df2))

df_main = pd.concat([df1, df2])
df_main.reset_index(drop=True)

# print(np.shape(df_main))

print(df_main.tail())