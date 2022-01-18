from multitracker.experiments import experiment_a, experiment_b, experiment_c, experiment_d, experiment_e, experiment_f, experiment_g


def all_experiments(args):
    max_steps = 15000 
    #max_steps = 2500
    #experiment_a.experiment_a(args, max_steps) # data ratios kp
    #experiment_b.experiment_b(args, max_steps) # random init kp
    #experiment_c.experiment_c(args, max_steps) # different backbones kp
    #experiment_e.experiment_e(args, train_video_ids=args.train_video_ids) # data ratios ob
    #experiment_f.experiment_f(args, train_video_ids=args.train_video_ids) # ssd ob
    #experiment_d.experiment_d(args, max_steps, train_video_ids=args.train_video_ids) # different train losses kp
    experiment_g.experiment_g(args, train_video_ids=args.train_video_ids) # pretrained augmentations ob

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--test_video_ids',required=True,type=str)
    parser.add_argument('--train_video_ids',required=True,type=str)
    args = parser.parse_args()
    all_experiments(args)