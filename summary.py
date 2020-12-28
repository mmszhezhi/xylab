import os
import re


def search(root,targets,reuslt,exclude=None,prefix=None):
    if not isinstance(targets,list):
        targets = [targets]
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            # print('[-]', path)
            search(path, targets,reuslt,exclude,prefix)
        # elif item.find(target) != -1:
        #     print('[+]', path)
        elif any([re.search(target,item) for target in targets]):
            if exclude and item.endswith(exclude):
                continue
            if prefix and item.find(prefix) == -1:
                continue
            print('[+]', path)
            reuslt.append(path)

# search("D:\d","test")
reuslt = []
search(r"D:\\",["T03","T0999"],reuslt,exclude=".png",prefix="load")
print(f"find {len(reuslt)}")
