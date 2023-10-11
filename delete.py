import os

path = "Speed_Violents"

for data in os.listdir(path):
    data_choose = os.path.join(path, data)
    if os.path.isfile(data_choose):
        os.unlink(data_choose)