import argparse
import re, math

import matplotlib as mpl
import matplotlib.pyplot as plt

def get_runwise_data(output_file):
    """Get data in form of list of scores, and their standard deviations 
    from an output file of a run."""
    with open(output_file) as f:
        data = f.read().split('Train size')[args.split]
                    
    run_epoch_wise_data = []
    for _run_no, run_data in enumerate(data.split('RUN')[1:]):
        run_epoch_wise_data.append([])
        
        splitted_run_data = run_data.split('EPOCH')[1:]
        if len(splitted_run_data) <= 1:
            splitted_run_data = run_data.split('Epoch')[1:]
        
        for epoch_no, epoch_data in enumerate(splitted_run_data):
            try:
                if args.dotall:
                    regex = re.compile(args.regexp, re.DOTALL)
                else:
                    regex = re.compile(args.regexp)

                f1 = re.findall(regex, epoch_data)[0]
                print(f1)
                f1 = int(f1)/(10**len(f1))
                run_epoch_wise_data[-1].append(f1)
            
            except IndexError:
                if epoch_no!=len(run_data.split('EPOCH')[1:])-1:
                    run_epoch_wise_data[-1].append(0.0)
                else:
                    print("Error: Could not find any score in output file, for the provided regular expression")
                    exit(1)
                
    for elem in run_epoch_wise_data:
        print("Read data for:", len(elem), "epochs.")
    
    n_epochs = min([len(elem) for elem in run_epoch_wise_data])
    print("Finally considering data upto:", n_epochs, "epochs.")

    return [[run_wise_data[i] for run_wise_data in run_epoch_wise_data]
            for i in range(n_epochs)]

def mean(lis):
    return sum(lis)/len(lis)

def bessel_corrected_std(lis):
    m = mean(lis)
    variances = [(elem-m)*(elem-m) for elem in lis]
    return 0 if len(lis)<=1 else math.sqrt(sum(variances)/(len(lis)-1))

def get_mean_and_error(epoch_run_wise_data):
    means = []
    stds = []
    for epoch_data in epoch_run_wise_data:
        means.append(mean(epoch_data))
        stds.append(bessel_corrected_std(epoch_data))
    return means, stds

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Plots the apt score for all the input files onto a single graph.")
    parser.add_argument("--in_files", type=str, nargs="+", required=True, help="Input files containing data to plot.")
    parser.add_argument("--names", type=str, nargs="+", required=True, help="Names for various plots of various files.")
    parser.add_argument("--avg_over", type=int, default=5, help = "The mean scores printed are averaged over this many last scores.")
    parser.add_argument("--regexp", type=str, required=True, help="Regular expression to search for scores. ")
    parser.add_argument("--split", type=int, default=1, help="Index of split to choose, after splitting the data in a file on \"Train size\"")
    parser.add_argument("--dotall", action="store_true", help="Use the dotall flag in the regular expression.")
    args = parser.parse_args()

    means = []
    stds = []
    for filename in args.in_files:
        mean_lis, std_lis = get_mean_and_error(get_runwise_data(filename))
        means.append(mean_lis)
        stds.append(std_lis)
    
    if args.avg_over>=len(means[0]):
        raise ValueError("Not enough epochs to average over. Kindly decrease the epochs to \
                          average over via --avg_over. No. of epochs:", len(means), 
                          "Averaging over:", args.avg_over)
    
    print("Mean scores for files provided, over the last", args.avg_over,"runs, in order:")
    
    for mean_lis, name in zip(means, args.names):
        print("\t", "Score for", name, ":", mean(mean_lis[-args.avg_over:]))
