from glob import glob

dir_A = "../datasets/MTandHE/" 

# AorB = "A"
paths = []
paths.append(dir_A +"train/*png")
paths.append(dir_A +"test/*png")

pic_namelist =[]
for path in paths:
    pic_namelist += glob(path)

from PIL import Image

for A_path in pic_namelist:
    # print(A_path)
    A_img = Image.open(A_path).convert('RGB')
    w,h = A_img.size
    if(w != 2048):
        print(w)
        print(A_path)
    if(h != 512):
        print(h)
        print(A_path)
    