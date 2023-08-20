from argparse import ArgumentParser
from plot import plot
from train import train

def main() :
    valid_subjects = list(range(1, 110))  # Subjects from 1 to 109
    valid_experiments = list(range(3, 15))
    func_dict = {
        "plot" : plot,
        "train" : train
    }
    parser = ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    plot_parser = sub_parser.add_parser("plot")
    plot_parser.add_argument("subject", type=int, choices=valid_subjects, help="Subject ID")
    plot_parser.add_argument("experiment", type=int, choices=valid_experiments, help="Experiment ID")
    train_parser = sub_parser.add_parser("train")
    train_parser.add_argument("subject", type=int, choices=valid_subjects, help="Subject ID")
    train_parser.add_argument("experiment", type=int, choices=valid_experiments, help="Experiment ID")
    args = parser.parse_args()
#    try :
    command = args.command
    args = vars(args)
    args.pop("command")
    func_dict[command](**args)
#    except Exception as e :
#        print(str(e))
#        parser.print_help()

if __name__ == "__main__" :
    main()