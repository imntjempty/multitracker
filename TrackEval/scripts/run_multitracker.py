
""" run_multitracker.py

Run example:
run_bdd.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL qdtrack

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
            'GT_FOLDER': os.path.join(code_path, 'data/gt/bdd100k/bdd100k_val'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/bdd100k/bdd100k_val'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle'],
            # Valid: ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']
            'SPLIT_TO_EVAL': 'val',  # Valid: 'training', 'val',
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    Metric arguments:
        'METRICS': ['Hota','Clear', 'ID', 'Count']
"""

from copy import deepcopy
import sys
import os
import argparse
from multiprocessing import freeze_support
import json 
import numpy as np 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def do_evaluate():
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_dataset_config = trackeval.datasets.BDD100K.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    parser.add_argument('--json_config', default = '~/github/multitracker/TrackEval/configs/TrackingUnderOcclusion.json')
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if setting not in config:
                config[setting] = args[setting]
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
            
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    if 'json_config' in config and config['json_config'] is not None:
        with open(os.path.expanduser(config['json_config'])) as fjson:
            user_config = json.load(fjson)
    
    for sequence in user_config["eval_jobs"]:
        print('[*] starting evaluation for sequence', sequence['sequence_name'])

        
        _dataset_config = deepcopy(dataset_config)
        _dataset_config['_csv_trackannotation'] = sequence['csv_trackannotation']
        
        for tracker in sequence['trackers']:
            __dataset_config = deepcopy(_dataset_config)
            __dataset_config['_csv_tracked'] = tracker['csv_out']
            
            # Run code
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.Multitracker(__dataset_config)]
            metrics_list = []
            for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
                if metric.get_name() in metrics_config['METRICS']:
                    metrics_list.append(metric())
            if len(metrics_list) == 0:
                raise Exception('No metrics selected for evaluation')
        
            output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
            #print('output_res\n\n\n')
            #print(output_res)
            output_res['sequence_name'] = sequence['sequence_name']
            output_res['tracker_name'] = tracker['name']
            json_out = '%s_%s.json' % ( sequence['sequence_name'], tracker['name'])
            with open(json_out, 'w') as outfile:
                json.dump(output_res, outfile, cls=NumpyEncoder)


def plot_eval_results():
    """ 
        write csv with table of results

                 
        Sequence | Tracker  | HOTA, MOTA, IDF1, DetA, AssA, DetRe, DetPr, AssRe, AssPr, IDSW
        seg1     | DeepSort |
                 | V-IoU    |
                 | UBT      |
        seg2     | DeepSort |
                 | V-IoU    |
                 | UBT      |
        avg      | DeepSort |
                 | V-IoU    |
                 | UBT      |

        write bar plot where 
    """


if __name__ == '__main__':
    do_evaluate()
    plot_eval_results()