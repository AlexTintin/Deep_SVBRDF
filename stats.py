from utils import config
import glob, os
import pandas as pd

config = config()

data_dico = {}


os.chdir(config.path.data_path)
for index_file, file in enumerate(glob.glob("*.png")):
    print(file.split(";"))
    split_name = file.split(";")
    data_dico[int(split_name[0])] = {'nom_file':file, "id":int(split_name[0]),
                                     "texture":split_name[1].split(("_")), "info_inconnue": split_name[1].split(("."))[0] }

df = pd.DataFrame(data_dico)
hist = df.hist()
