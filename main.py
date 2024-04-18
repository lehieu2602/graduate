from graph.network import *
from graph.sfc_set import *
from NSGAII import *

if __name__ == '__main__':
    network = Network(input_path=r"D:\PY\graduate\data_with_delay_v2\input00\input.txt")
    # print(network.__repr__())
    SFCs = SFC_SET(input_path=r"D:\PY\graduate\data_with_delay_v2\input00\request10.txt")
    # print(SFCs.__repr__())
    # network.visualize()
    set_value_network(network)
    set_SFCs(SFCs)
    main(20, 200, 90)
