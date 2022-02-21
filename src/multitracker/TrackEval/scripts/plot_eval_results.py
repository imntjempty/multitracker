"""
    plot eval results

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os 

def main(config):
    dataframe = pd.read_csv(config['csv'])
    os.makedirs(config['out'], exist_ok=True)

    for metric_name in ['IDSW','IDF1','IDTP','IDFN','IDFP','HOTA(0)']:
        fig, ax = plt.subplots()
        ax.set_title(metric_name)
        ax.scatter(dataframe['tracker'], dataframe[metric_name], c='g')
        #plt.show()  # or plt.savefig("name.png")
        plt.tight_layout()
        plt.savefig(os.path.join(config['out'],'%s.png' % metric_name))
        plt.close('all')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--out')
    args = vars(parser.parse_args())
    args['csv'] = os.path.expanduser(args['csv'])
    args['out'] = os.path.expanduser(args['out'])
    main(args)