import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

#Data files are available in the folder of the project
#For example, running this (by clicking run or pressing Shift+Enter) will list the files in the project directory

from subprocess import check_output
#print(check_output(["ls", "../folder"]).decode("utf8"))
df = pd.read_csv(".../Financial Distress.csv")
df.head()
# Any results you write to the current directory are saved as output.