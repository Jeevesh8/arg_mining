#Script to first make OP-wise split of the CMV Modes data and then choose train-test-valid set from it
import argparse
import os, random
from shutil import copyfile, rmtree
from typing import Dict

from bs4 import BeautifulSoup

def get_thread_ids_to_filenames(root_folder="./v2.0/"):
    thread_ids_to_filenames= {}
    for t in ["negative", "positive"]:
        root = root_folder + t + "/"
        for f in os.listdir(root):
            filename = os.path.join(root, f)
            if os.path.isfile(filename) and f.endswith(".xml"):
                with open(filename, "r") as g:
                    xml_str = g.read()
                parsed_xml = BeautifulSoup(xml_str, "xml")
                thread_id = parsed_xml.find('thread')['ID']
                if thread_id not in thread_ids_to_filenames.keys():
                    thread_ids_to_filenames[thread_id] = []
                thread_ids_to_filenames[thread_id].append(filename)

    with open('./op_wise_split.txt', 'w+') as f:
        for v in thread_ids_to_filenames.values():
            f.write(str(v)+'\n')
    
    return thread_ids_to_filenames

def main(args) -> Dict[str, str]:
    out_folders = {}
    
    assert args.train_sz + args.test_sz == 100, "Train and test size should sum to 100."

    if args.save_folder=="":
        args.save_folder = os.path.join(args.data_folder, "data/")
    
    thread_ids_to_filenames = get_thread_ids_to_filenames(args.data_folder)
    
    for split in ["train", "test", "valid"]:
        if os.path.isdir(os.path.join(args.save_folder, split)):
            rmtree(os.path.join(args.save_folder, split))
        os.makedirs(os.path.join(args.save_folder, split), exist_ok=True)
        out_folders[split] = os.path.join(args.save_folder, split)
    

    op_wise_threads_lis = [elem for elem in thread_ids_to_filenames.values()]
    if args.shuffle:
        random.shuffle(op_wise_threads_lis)
    
    total_threads = sum([len(elem) for elem in op_wise_threads_lis])
    
    j=0    
    for elem in op_wise_threads_lis:
        if 100*(j/total_threads)<args.test_sz:
            for filename in elem:
                copyfile(filename, os.path.join(args.save_folder, "test", filename.split('/')[-2]+'_'+filename.split('/')[-1]))
                j+=1
        else:
            for filename in elem:
                copyfile(filename, os.path.join(args.save_folder, "train", filename.split('/')[-2]+'_'+filename.split('/')[-1]))
                j+=1
    
    #Make dummy validation folder
    copyfile(filename, os.path.join(args.save_folder, "valid", filename.split('/')[-2]+'_'+filename.split('/')[-1]))
    return out_folders

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sz", required=True, type=float, help="Percentags of threads to use for train set.")
    parser.add_argument("--test_sz", required=True, type=float, help="Percentage of threds in test set.")
    parser.add_argument("--data_folder", default="./", help="The folder that has negative/ and positive/ subfolders having threads.")
    parser.add_argument("--save_folder", default="", help="Folder to store final data.")
    parser.add_argument("--shuffle", action="store_true", help="If this flag is provided, data is shuffled before making train-test split. \
                                                                Hence giving a random train-test split.")
    return parser

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)