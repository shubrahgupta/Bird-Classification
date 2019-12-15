#import the libraries
import pandas as pd
import os
from pydub import AudioSegment

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
    path=path1+"/"+str(b[i])+"/finally1.csv"
    e=d.to_csv(path)  #exporting the final csv into their respective folders


#audiofile location
files_path = '/home/shubrah/Downloads/'
file_name = 'P03929TEA1_20170506_070001'


# Opening file and extracting segment
for i in range(len(b)):
    d=c.get_group(b[i])
    for j in range(len(d)):
        start=d["Begin Time (s)"].values[j]*1000  #starttime
        end=d["End Time (s)"].values[j]*1000     #endtime
        song = AudioSegment.from_wav( files_path+file_name+'.wav' ) #extracting song
        extract = song[start:end]   #cutting the song
        path=path1+"/"+str(b[i])+"/"    #path for exporting
        extract.export(path+'extract.wav', format="wav") #saving the file

