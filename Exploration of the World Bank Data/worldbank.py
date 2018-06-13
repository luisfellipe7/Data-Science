# importing libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
%matplotlib inline
pd.options.display.max_columns = 50

# importing datasets
df1 = pd.read_csv('../input/bilateral-remittance.csv')
df2 = pd.read_csv('../input/remittance-inflow.csv')
df3 = pd.read_csv('../input/remittance-outflow.csv')

# seeing the raw data we have
df2.head(10)

#processing data
df2 = df2.iloc[:214,1:]
df2.index=df2['Migrant remittance inflows (US$ million)']
df2.index.names = ['Country']
df2.columns.names = ['Year']
df2 = df2.drop('Migrant remittance inflows (US$ million)', axis=1)
df2 = df2.dropna(axis=0, how='all')

#now we calculate the syn if the inflow in the past years by each country
df2['Sum of Inflow Money'] = np.zeros(len(df2))
df2 = df2.astype(float)
for i in range(len(df2)):
    df2['Sum of Inflow Money'][i] = df2.iloc[i].sum()
df2.sort_values(by = 'Sum of Inflow Money', ascending=False, inplace=True)
df2['Sum of Inflow Money'][:10]

#to visualize the sorted value
fig = plt.figure(figsize=(30,10))
plt.plot(df2['Sum of Inflow Money'].values, label = 'Sum of Inflow Money')
plt.xlabel('Country')
labels = df2.index.values
x = range(len(df2.index))
plt.xticks(x, labels, rotation = 'vertical')
plt.ylabel('US$ million')
plt.title('Sum of Inflow Money by Countries',fontsize = 40)
plt.show()

#another raw data
df3.head(10)

#processing
df3 = df3.iloc[:214,1:]
df3.index= df3['Migrant remittance outflows (US$ million)']
df3.index.names = ['Country']
df3.columns.names = ['Year']
df3 = df3.drop('Migrant remittance outflows (US$ million)', axis=1)
df3 = df3.dropna(axis=0, how='all')

# Calculate the sum of outflow in the past years by country
df3['Sum of Outflow Money'] = np.zeros(len(df3))
df3 = df3.astype(float)
for i in range(len(df3)):
    df3['Sum of Outflow Money'][i] = df3.iloc[i].sum()
df3.sort_values(by = 'Sum of Outflow Money', ascending=False, inplace=True)
df3['Sum of Outflow Money'][:10]

# Visualizing the sorted value
fig = plt.figure(figsize=(30,10))
plt.plot(df3['Sum of Outflow Money'].values, label = 'Sum of Outflow Money')
plt.xlabel('Country')
labels = df3.index.values
x = range(len(df3.index))
plt.xticks(x, labels, rotation = 'vertical')
plt.ylabel('US$ million')
plt.title('Sum of Outflow Money by Countries',fontsize = 40)
plt.show()

#another raw data
df1.head(10)

# Data preprocessing
df1 = df1.iloc[:214,1:-1]
df1.index= df1[df1.columns[0]]
df1.index.name = None
df1 = df1.drop(df1.columns[0], axis=1)
for i in range(len(df1.columns)):
    for j in range(len(df1)):
        df1.iloc[j,i] = re.findall(r"\d+\.?\d*", str(df1.iloc[j,i]))[0]
df1 = df1.astype(float)

# Visualizing the heat map of money flow of 2016
f, ax = plt.subplots(figsize=(40, 40))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
vis = sns.heatmap(df1, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Calculating the 2016 flow money by country
df1['Sum of 2016 Flow Money'] = np.zeros(len(df1))
for i in range(len(df1)):
    df1['Sum of 2016 Flow Money'][i] = df1.iloc[i].sum()
df1.sort_values(by = 'Sum of 2016 Flow Money', ascending=False, inplace=True)
df1['Sum of 2016 Flow Money'][:10]

# Visualizing the sorted value
fig = plt.figure(figsize=(30,10))
plt.plot(df1['Sum of 2016 Flow Money'].values, label = 'Sum of 2016 Flow Money')
plt.xlabel('Country')
labels = df1.index.values
x = range(len(df1.index))
plt.xticks(x, labels, rotation = 'vertical')
plt.ylabel('US$ million')
plt.title('Sum of 2016 Flow Money by Countries',fontsize = 40)
plt.show()
