import pandas as pd
import os
a=pd.read_csv('/home/shubrah/Downloads/new.txt',sep="\t")
a["Diff"]=a["End Time (s)"]-a["Begin Time (s)"]
b=list(set(a["Notes"]))

# define the name of the directory to be created
path1 = "/home/shubrah/bird/"

# define the access rights
access_rights = 0o755
try:
    os.mkdir(path1,access_rights)    
except FileExistsError:
    print("Directory already exists")


for i in range(len(b)):
    path=path1+str(b[i])
    try:
        os.mkdir(path,access_rights)    
    except FileExistsError:
        print("Directory already exists")
        continue

c=a.groupby("Notes")
for i in range(len(b)):
    d=c.get_group(b[i])
    path=path1+str(b[i])+"/finally.csv"
    e=d.to_csv(path)
    
