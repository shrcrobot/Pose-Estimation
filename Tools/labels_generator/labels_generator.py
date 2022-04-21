import os.path

cls = {}
cnt = 0
name_file = open("./raw/pcd_names_file.txt","w")
label_file = open("./raw/labels_file.txt","w")

for i in os.listdir("./raw/"):
    filename=os.path.splitext(i)
    if filename[1]==".pcd":
        if filename[0].split('_')[0] not in cls:
            cls[filename[0].split('_')[0]]=cnt
            cnt=cnt+1
        name_file.writelines(i+"\n")
        label_file.writelines(str(cls[filename[0].split('_')[0]])+"\n")

name_file.close()
label_file.close()