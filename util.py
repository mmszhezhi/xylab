import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob,os
import json
from filereader import filter,base_filter,search

def file_read(file,src):
    with open(os.path.join(src,file)) as f:
        data = f.read()
    data = json.loads(data)
    return data["value"]


def draw_image(files,src,dst):
    p = 1
    for item in files:
        try:
            df = None
            cat = os.path.basename(item)
            i = cat.split("_")[2]
            if i in [str(15),str(11),str(4),str(50),str(127)]:
                print("ff")
            print(f"{p} of {len(file2draw)}")
            p += 1
            temp = {}
            temp["load"] = file_read(item,src)
            for k,v in channel_names.items():
                if v == "sound":
                    continue
                temp[v] = file_read(item.replace("load",v),src)
            fig,ax = plt.subplots(4,1,figsize=(25,13))
            for j,(k,v) in enumerate(temp.items()):
                ax[j].plot(v)
                patch = mpatches.Patch(color='blue', label=k)
                ax[j].legend([k],loc="upper left")
            # l = plt.legend(loc=2)
            df,s,e = base_filter(temp["load"],i)
            if df is not None:
                ax[0].scatter(x=df["index"],y=temp["load"],c=df["label"])
            plt.savefig(os.path.join(dst,item + ".png"))
            plt.close()

        except Exception as e:
            print(repr(e))

def draw_image_vib(files,src,dst,ch=('1',"2","3"),load=False):
    p = 1
    for item in files:
        try:
            df = None
            cat = os.path.basename(item)
            i = cat.split("_")[2]
            if i in [str(15),str(11),str(4),str(50),str(127)]:
                print("ff")
            print(f"{p} of {len(file2draw)}")
            p += 1
            temp = {}
            if load:
                temp["load"] = file_read(item,src)
            for k in ch:
                v = channel_names.get(k)
                if not v:
                    continue
                temp[v] = file_read(item.replace("load",v),src)
            fig,ax = plt.subplots(len(ch),1,figsize=(25,13))
            for j,(k,v) in enumerate(temp.items()):
                ax[j].plot(v)
                patch = mpatches.Patch(color='blue', label=k)
                ax[j].legend([k],loc="upper left")
            # l = plt.legend(loc=2)
            if load:
                df,s,e = base_filter(temp["load"],i)
                if df is not None:
                    ax[0].scatter(x=df["index"],y=temp["load"],c=df["label"])
            plt.savefig(os.path.join(dst,item + ".png"))
            plt.close()

        except Exception as e:
            print(repr(e))

def retrive_recursive(src,dst):
    s1 = set()
    s2 = set()
    for item in glob.glob(src + "/*"):
        filename = os.path.basename(item)
        dirname = os.path.dirname(item)
        try:
            if "load" == filename.split("_")[1]:
                s1.add(filename)
        except:
            pass
    ip = os.path.join(src, dst)
    os.makedirs(ip,exist_ok=True)
    for item in glob.glob(ip+"/*"):
        filename = os.path.basename(item)
        filename = filename.replace(".png","")
        s2.add(filename)
    dif = s1.difference(s2)
    return dif,dirname,ip

# root = r"D:\xy\data\20201214\192.168.101.10"
# root = r"D:\xy-repo\data\20201214\192.168.101.10"
root = r"D:\xy-repo\data\20201214\192.168.101.10"
cuts = ["T03"]
channel_names = {'1': 'vibration-x', '2': 'vibration-y', '3': 'vibration-z', '4': 'sound'}
if __name__ == '__main__':
    for c in cuts:
        print(c)
        base = os.path.join(root,c)
        file2draw,src,dst = retrive_recursive(base,"imagess")
        draw_image(file2draw,src,dst)
        # draw_image_vib(file2draw, src, dst)
