import argparse
import re, math

import matplotlib as mpl
import matplotlib.pyplot as plt

def get_runwise_data(output_file):
    """Get data in form of list of scores, and their standard deviations 
    from an output file of a run."""
    split = 1 if "80-20" in args.title else 2
    with open(output_file) as f:
        data = f.read().split('Train size')[split]
                    
    run_epoch_wise_data = []
    for _run_no, run_data in enumerate(data.split('RUN')[1:]):
        run_epoch_wise_data.append([])
        for epoch_no, epoch_data in enumerate(run_data.split('EPOCH')[1:]):
            
            try:
                if 'krippendorff alpha' in args.title.lower():
                    if "claim" in args.title.lower():
                        f1 = re.findall(r"Sentence level Krippendorff's alpha for Claims:  0.(\d\d\d|\d\d)", epoch_data)[0]
                    elif "premise" in args.title.lower():
                        f1 = re.findall(r"Sentence level Krippendorff's alpha for Premises:  0.(\d\d\d|\d\d)", epoch_data)[0]
                    else:
                      print("Neither Claim nor Premise found in title. Please specify one.")
                      exit(1)
                elif 'relation' in args.title.lower():
                    f1 = re.findall(r'\'weighted_avg\'.*\'f1\': 0.(\d\d\d|\d\d)', epoch_data)[0]
                else:
                    f1 = re.findall(r'overall_f1.*: 0.(\d\d\d|\d\d)', epoch_data)[0]
                print(f1)
                f1 = int(f1)/(10**len(f1))
                run_epoch_wise_data[-1].append(f1)
            
            except IndexError:
                if epoch_no!=len(run_data.split('EPOCH')[1:])-1:
                    run_epoch_wise_data[-1].append(0.0)
                else:
                    print("Error: Could not find F1 score in output file.")
                    exit()
                
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
    parser.add_argument("--out_file", type=str, required=True, help="Output file to save the plot to.")
    parser.add_argument("--avg_over", type=int, default=5, help = "The mean scores printed are averaged over this many last scores.")
    parser.add_argument("--title", type=str, default="", help="Title of the plot.", required=True)
    parser.add_argument("--human_perf", type=float, default=0.0, help="If provided, human performance is also plotted.")
    parser.add_argument("--majority_class", type=float, default=0.0, help="If provided, the majority class is also plotted.")
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

    fig, ax = plt.subplots()
    
    for mean_lis, label in zip(means, args.names):
        ax.plot(mean_lis, label=label)
    
    if args.human_perf != 0:
        ax.plot([args.human_perf]*len(means[0]), label="Human Performance")
    if args.majority_class != 0:
        ax.plot([args.majority_class]*len(means[0]), label="Majority Class")
    
    ax.legend(loc='lower right')

    for mean_lis, std_lis in zip(means, stds):
        plt.fill_between([i for i in range(len(mean_lis))],
                         [elem1-elem2 for elem1, elem2 in zip(mean_lis, std_lis)], 
                         [elem1+elem2 for elem1, elem2 in zip(mean_lis, std_lis)],
                         alpha=0.4, interpolate=False)
    plt.xlabel("Epochs")
    
    if 'krippendorff alpha' in args.title.lower():
        plt.ylabel("Sentence Level Krippendorff Alpha")
    elif 'relation' in args.title.lower():
        plt.ylabel("Weighted f1")
    else:
        plt.ylabel("Overall f1")
    
    plt.title(args.title)

    plt.savefig(args.out_file)