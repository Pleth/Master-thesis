import sys
import matplotlib.pyplot as plt
import numpy as np

from functions import *
from LiRA_functions import *


if __name__ == '__main__':
    
    # if len(sys.argv) != 4:
    #     print("3 arguments needed")
    #     sys.exit(1)
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]
    # arg3 = sys.argv[3]
    
    
    # # arg1 - Train or eval
    # if arg1 == "SVM":
    #     print(arg1)
    # elif arg1 == "KNN":
    #     print(arg1)
    # elif arg1 == "DT":
    #     print(arg1)
    # else:
    #     print("Choose argument 1 (Method): SVM, KNN or DT")
    #     sys.exit(1)
    
    # # arg2 - Train or eval
    # if arg2 == "train":
    #     print(arg2)
    # elif arg2 == "eval":
    #     print(arg2)
    # else:
    #     print("Choose argument 2 (Training or evaluation): train or eval") 
    #     sys.exit(1)   

    # # arg3 - Data
    # if arg3 == "real":
    #     print(arg3)
    # elif arg3 == "sim":
    #     print(arg3)
    # else:
    #     print("Choose argument 3 (Real data or simulated data): real or sim")
    #     sys.exit(1)
    

    # #Synth data
    # file_p79 = "tosend/data/p79_data.csv"
    # file_gm = "tosend/data/green_mob_data.csv"

    # df_gm = pd.read_csv(file_gm) # speed_gm,times
    # df_p79 = pd.read_csv(file_p79) #distances,laser5,laser21

    # synth_data = synthetic_data = create_synthetic_signal(
    #                         p79_distances=np.array(df_p79["distances"]),
    #                         p79_laser5=np.array(df_p79["laser5"]),
    #                         p79_laser21=np.array(df_p79["laser21"]),
    #                         gm_times=np.array(df_gm["times"]),
    #                         gm_speed=np.array(df_gm["speed_gm"]))

    # synth_acc = pd.DataFrame(synth_data["synth_acc"],columns = ['synth_acc'])
    # #plt.plot(synth_data["times"],synth_data["synth_acc"])
    # #plt.show()
    # synth_acc.to_csv("synth_data/synth.csv",index=False)
    synth_acc = synthetic_data()
    
    out = pd.DataFrame(data_window(synth_acc))
    out.insert(0,"id",np.ones(125,))

    from tsfresh import extract_features, extract_relevant_features
    extracted_features = extract_features(out.iloc[:,0:2],column_id="id")


