import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description="Plot the variation of percentage of erroroneous\
                                                  predictions with distance between components.")
    
    parser.add_argument("--input_file", type=str, required=True, help="Input file with the predictions.")
    parser.add_argument("--epoch_no", type=int, required=True, help="Epoch number whose predictions are \
                                                                     to be considered.")
    parser.add_argument("--bin_size", type=int, default=-1, help="Bin size for the histogram. If not provided, \
                                                                  bin sizes will be adjusted automatically to make \
                                                                  sure equal number of predictions lie in each bin.")
    parser.add_argument("--num_bins", type=int, default=-1, help="Number of bins to make for the histogram.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the plot.")    
    return parser

def read_file(file_name):
    preds, actual, dist = [], [], []
    with open(file_name, 'r') as f:
        log_data = f.read()
    epoch_data = log_data.split("EPOCH")[args.epoch_no]
    for line in epoch_data.split("\n")[1:]:
        print(line)
        if line.startswith("Evaluating"):
            continue
        elif line.startswith("Predicted"):
            preds.append(line.split(":")[-1].strip())
        elif line.startswith("Actual"):
            actual.append(line.split(":")[-1].strip())
        elif line.startswith("Tokens between"):
            dist.append(int(line.split(":")[-1].strip()))
        else:
            break
    assert len(preds) == len(actual) == len(dist) 
    return [elem for elem in zip(preds, actual, dist)]

def split_by_bin_size(data_lis: List[Tuple[str, str, int]]) -> List[List[Tuple[str, str, int]]]:
    """Splits the data_lis according to which bin(of size bin_size) the dist of 
    current tuple should lie in.
    Args:
        data_lis:   List of tuples of the form (pred, actual, dist)
    Returns:
        A list having the same number of elements as the number of bins. Each element 
        is a list of tuples of the form (pred, actual, bin_no). 
    """
    bins = {}
    for pred, actual, dist in data_lis:
        bin_no = dist // args.bin_size
        if bin_no not in bins:
            bins[bin_no] = []
        bins[bin_no].append((pred, actual, bin_no))
    return [bins[bin_no] for bin_no in sorted(bins.key())]

def split_by_num_bins(data_lis: List[Tuple[str, str, int]]) -> Tuple[List[int], List[Tuple[str, str, int]]]:
    """Splits the data_lis into args.num_bins bins of equal size, and return the corresponding
    boundaries obtained.
    Args:
        data_lis:   List of tuples of the form (pred, actual, dist)
    Returns:
        Two lists:
            1. List of the boundaries of the bins.
            2. List of List of tuples of the form (pred, actual, dist); where the inner list consists of
            all elements in a single bin.
    """
    sorted_data_lis = sorted(data_lis, key=lambda x: x[2])
    bin_boundaries = []
    bin_contents = []
    elements_per_bin = len(data_lis) // args.num_bins
    for i in range(args.num_bins):
        start_idx = i * elements_per_bin
        end_idx = (i + 1) * elements_per_bin
        bin_boundaries.append(sorted_data_lis[start_idx][2])
        bin_contents.append(sorted_data_lis[start_idx:end_idx])
    return bin_boundaries, bin_contents

def percentage_error(bin_content: List[Tuple[str, str, int]]) -> float:
    """Returns the percentage error for the predictions and actual values 
    of the bin, whose data is provided.
    """
    assert len(bin_content) > 0
    total = 0
    incorrect = 0
    for pred, actual, _ in bin_content:
        total += 1
        if pred != actual:
            incorrect += 1
    return 100*(incorrect / total)

def plot(xs: List[int], heights: List[float]):
    plt.step(xs, heights, where="post")
    plt.plot(xs, heights, 'o--', color="grey", alpha=0.3)
    plt.xlabel("Number of Tokens between components")
    plt.ylabel("Percentage of erroroneous predictions")
    plt.savefig(args.output_file)

def main():
    data_lis = read_file(args.input_file)
    if args.bin_size > 0:
        heights = []
        xs = []
        bin_contents = split_by_bin_size(data_lis)
        max_bin_no = max([bin[0][2] for bin in bin_contents])
        for i in range(1, max_bin_no+1):
            bin = bin_contents[i]
            if bin[0][2]+1!=i:
                heights.append(0)
            else:
                heights.append(percentage_error(bin))
            xs.append((i-1)*args.bin_size)
    
    elif args.num_bins > 0:
        bin_boundaries, bin_contents = split_by_num_bins(data_lis)
        heights = [percentage_error(bin) for bin in bin_contents]
        xs = bin_boundaries
    else:
        raise ValueError("Either bin_size or num_bins must be >0.")
    plot(xs, heights)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.bin_size == -1 and args.num_bins == -1:
        print("Please provide either bin size or number of bins.")
        exit(1)
    elif args.bin_size != -1 and args.num_bins != -1:
        print("Please provide only one of bin size or number of bins.")
        exit(1)
    main()