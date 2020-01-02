#import the libraries
import pandas as pd
import os
import uuid
import subprocess

#Extract the text file as csv 
a=pd.read_csv('/home/shubrah/Downloads/new1.txt',sep="\t")

a["Diff"]=a["End Time (s)"]-a["Begin Time (s)"]
#Created a new column for time difference

b=list(set(a["Notes"]))
#created a set of all species,no repetition

# define the name and location of the directory to be created
path1 = "/home/shubrah/bird/"

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
    x=d.drop(['Selection', 'View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)', 'Q1 Freq (Hz)', 'Q3 Freq (Hz)',
       'Avg Power (dB)', 'Begin Date', 'Delta Time (s)', 'Begin File',
       'Begin Path', 'Center Freq (Hz)', 'Delta Freq (Hz)'],axis=1)
    path=path1+"/"+str(b[i])+"/"+str(uuid.uuid4())+".csv"
    e=x.to_csv(path)  #exporting the final csv into their respective folders



#audiofile location
files_path = '/home/shubrah/Downloads/'
file_name = 'P03929TEA1_20170506_070001.wav'


# Opening file and extracting segment
for i in range(len(b)):
    d=c.get_group(b[i])
    for j in range(len(d)):
        start=str(d["Begin Time (s)"].values[j])  #starttime
        end=str(d["End Time (s)"].values[j])   #endtime
        path=path1+"/"+str(b[i])+"/"    
        final=path+str(uuid.uuid4())+str('.wav') #final path for file
        extracting_path=files_path+file_name  #path for exporting
        cmd=['ch_wave',extracting_path,'-o',final,'-start',start,'-end',end] #terminal_commands
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
