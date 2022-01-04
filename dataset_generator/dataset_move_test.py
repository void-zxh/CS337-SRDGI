import shutil
import os
import random
from tqdm import tqdm

raw_path="../output/"
class_list=["sce_test"]
root_path="../dataset/"
work_path=["test/"]
image_path=["albedo","depth","direct","gt","normal"]


def data_rename():
    for c in class_list:
        for w in work_path:
            for p in image_path:
                files=[]
                id=0
                path=root_path+c+'/'+w+p
                list_name=[]
                for file in os.listdir(path): 
                    list_name.append(file)
                list_name.sort()
                for file in list_name:
                    id=id+1
                    os.rename(os.path.join(path,file),os.path.join(path,str(id)+".png"))
                
def data_split():
    for c in class_list:
        path=raw_path+c
        image_list={}
        image_list["direct"]=[]
        image_list["albedo"]=[]
        image_list["depth"]=[]
        image_list["rgb"]=[]
        image_list["gt"]=[]
        image_list["normal"]=[]
        for file in os.listdir(path):
            if file.find("blend")!=-1:
                image_list["direct"].append(file)
            elif file.find("depth")!=-1:
                image_list["depth"].append(file)
            elif file.find("mat")!=-1:
                image_list["albedo"].append(file)
            elif file.find("normal")!=-1:
                image_list["normal"].append(file)
            elif file.find("rgb")!=-1:
                image_list["rgb"].append(file)
            else:
                image_list["gt"].append(file)
        print(len(image_list["direct"]))
        print(len(image_list["depth"]))
        print(len(image_list["normal"]))
        print(len(image_list["rgb"]))
        print(len(image_list["gt"]))
        #print(image_list["gt"])
        #print(image_list["direct"])
        for key in image_list.keys():
            image_list[key].sort()
        list_len=len(image_list["gt"])
        val_cnt = int(list_len/5)
        test_cnt = int(list_len/10)
        train_cnt = list_len - val_cnt - test_cnt
        # print("1:",val_cnt)
        # print("1:",test_cnt)
        # print("1:",train_cnt)
        data_list={}
        data_list["val/"]=[]
        data_list["test/"]=[]
        data_list["train/"]=[]
        for id in range(0,list_len):
            data_list["test/"].append(id)
        # print(len(data_list["val/"]))
        # print(len(data_list["test/"]))
        # print(len(data_list["train/"]))
        for w in work_path:
            for id in data_list[w]:
                for p in image_path:
                    dst_path=root_path+c+'/'+w+p
                    if not os.path.exists(dst_path):
                        os.makedirs(dst_path)
                    shutil.copy(path+"/"+image_list[p][id], dst_path)

data_split()
data_rename()