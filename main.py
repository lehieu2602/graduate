from graph.network import *
from graph.sfc_set import *
from NSGAII import *
import os

if __name__ == '__main__':
    folder_path = r"dataset"

    for direction in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, direction)

        files = os.listdir(dir_path)
        network_path = os.path.join(dir_path, files[0])
        network = Network(input_path=network_path)
        for file in files[1:]:
            sfc_path = os.path.join(dir_path, file)
            SFCs = SFC_SET(input_path=sfc_path)
            set_value_network(network)
            set_SFCs(SFCs)
            main(20, 50, 90)
