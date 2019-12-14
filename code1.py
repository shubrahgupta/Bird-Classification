#import the libraries
import pandas as pd
import os

#Extract the text file as csv 
a=pd.read_csv('<location>/new.txt',sep="\t")

a["Diff"]=a["End Time (s)"]-a["Begin Time (s)"]
#Created a new column for time difference

b=list(set(a["Notes"]))
#created a set of all species,no repetition

# define the name and location of the directory to be created
path1 = "<location>/bird/"

# define the access rights
access_rights = 0o755

#make bird directory
try:
    os.mkdir(path1,access_rights)    
except FileExistsError:
    print("Directory already exists")

#making directories
for i in range(len(b)):
    path=path1+"/"+str(b[i])
    try:
        os.mkdir(path,access_rights)    
    except FileExistsError:
        print("Directory already exists")
        continue

#grouping the dataframe according to the bird name code
c=a.groupby("Notes")
for i in range(len(b)):
    d=c.get_group(b[i])
    path=path1+"/"+str(b[i])+"/finally.csv"
    e=d.to_csv(path)  #exporting the final csv into their respective folders
