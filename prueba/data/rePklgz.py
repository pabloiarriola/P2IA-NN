#-*- coding: utf-8 -*-

from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd

data0 = "0\\*.png"
data1 = "1\\*.png"
data2 = "2\\*.png"
data3 = "3\\*.png"
data4 = "4\\*.png"
data5 = "5\\*.png"
data6 = "6\\*.png"
data7 = "7\\*.png"
data8 = "8\\*.png"
data9 = "9\\*.png"

# for windows, use this form, "bike_resize\\*.jpg" 


# def dir_to_dataset(glob_files, loc_train_labels=""):
def dir_to_dataset(glob_files):
    print("Process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        img = Image.open(file_name)
        #img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)

    return np.array(dataset)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    # if len(loc_train_labels) > 0:
    #     df = pd.read_csv(loc_train_labels)
    #     return np.array(dataset), np.array(df["Class"])
    # else:
    #     return np.array(dataset)

f = gzip.open('mnist.pkl.gz','wb')
Data = dir_to_dataset(data0)
np.random.shuffle(Data)
trn_size = 1341
val_size = 330
val_idx = trn_size+1+val_size
data_size = 2001
# Lee Data y Labels

train_set_y = list()
val_set_y = list()
test_set_y = list()

train_set_x = Data[:trn_size]
val_set_x = Data[trn_size+1:val_idx]
test_set_x = Data[val_idx+1:data_size]
for i in range(trn_size):
	train_set_y.append(0)
for i in range(val_size):
	val_set_y.append(0)
	test_set_y.append(0)


#DATOS 1	
	
Data = dir_to_dataset(data1)
# Data and labels are read 
np.random.shuffle(Data)
trn_size = 1341
val_size = 328
val_idx = trn_size+1+val_size
data_size = 1997
# Data and labels are read 

train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(1)
for i in range(val_size):
    val_set_y.append(1)
    test_set_y.append(1)

	
#DATOS 2	
	
Data = dir_to_dataset(data2)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 2001
val_size = 472
val_idx = trn_size+1+val_size
data_size = 2845
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(2)
for i in range(val_size):
    val_set_y.append(2)
    test_set_y.append(2)

	

Data = dir_to_dataset(data3)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 1341
val_size = 330
val_idx = trn_size+1+val_size
data_size = 2001
# Data and labels are read 

train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(3)
for i in range(val_size):
    val_set_y.append(3)
    test_set_y.append(3)

Data = dir_to_dataset(data4)
# Data and labels are read 
np.random.shuffle(Data)


trn_size = 1700
val_size = 391
val_idx = trn_size+1+val_size
data_size = 2482
# Data and labels are read 

train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(4)
for i in range(val_size):
    val_set_y.append(4)
    test_set_y.append(4)
	
	
	
	
#DATOS 6	
Data = dir_to_dataset(data5)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 2000
val_size = 525
val_idx = trn_size+1+val_size
data_size = 3050
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(5)
for i in range(val_size):
    val_set_y.append(5)
    test_set_y.append(5)	
	

	
#DATOS 7
	
Data = dir_to_dataset(data6)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 2344
val_size = 585
val_idx = trn_size+1+val_size
data_size = 3514
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(6)
for i in range(val_size):
    val_set_y.append(6)
    test_set_y.append(6)
	
	
#Datos 8

Data = dir_to_dataset(data7)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 1361
val_size = 330
val_idx = trn_size+1+val_size
data_size = 2021
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(7)
for i in range(val_size):
    val_set_y.append(7)
    test_set_y.append(7)
	

	
	
#DATOS 8
Data = dir_to_dataset(data8)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 1391
val_size = 330
val_idx = trn_size+1+val_size
data_size = 2051
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(8)
for i in range(val_size):
    val_set_y.append(8)
    test_set_y.append(8)
	
	
#Datos 9

Data = dir_to_dataset(data9)
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 1341
val_size = 330
val_idx = trn_size+1+val_size
data_size = 2001
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append(9)
for i in range(val_size):
    val_set_y.append(9)
    test_set_y.append(9)
	
	
	
	
	
	
	
	

train_set_x2 = list()
train_set_y2 = list()
val_set_x2 = list()
val_set_y2 = list()
test_set_x2 = list()
test_set_y2 = list()

print len(train_set_x)
arr = np.arange(len(train_set_x))
np.random.shuffle(arr)

for i in arr:
    train_set_x2.append(train_set_x[i])
    train_set_y2.append(train_set_y[i])

print "train shuffling completed...."
arr = np.arange(len(val_set_x))
np.random.shuffle(arr)

for i in arr:
    val_set_x2.append(val_set_x[i])
    val_set_y2.append(val_set_y[i])

print "validation shuffling completed...."
arr = np.arange(len(test_set_x))
np.random.shuffle(arr)

for i in arr:
    test_set_x2.append(test_set_x[i])
    test_set_y2.append(test_set_y[i])

print "test shuffling completed....."

train_set = train_set_x2, train_set_y2
val_set = train_set_x2, train_set_y2
test_set = test_set_x2, test_set_y2

dataset = [train_set, val_set, test_set]
print "dataset merged"
print "dumping...."
cPickle.dump(dataset, f, protocol=2)
print "completed!"
f.close()
