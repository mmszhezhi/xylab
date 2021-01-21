from model.models import DataGenerator
import json



if __name__ == '__main__':
    search_list = ("T07",'T09')

    datagen = DataGenerator(search_list=search_list, wins=30)
    datagen.init_source()
    datagen.cache_all()
    ret = {}
    for k,v in datagen.cache.items():
        ret[k] = v['sum'].to_list()
    with open('power79.json','w') as f:
        json.dump(ret,f)