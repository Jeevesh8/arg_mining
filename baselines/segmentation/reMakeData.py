import os, shlex, subprocess, argparse
import shutil, random
from pathlib import Path

import conll_data.make_data as make_data
import conll_data.xml_to_conll as xml_to_conll
import conll_data.conll_to_btds as conll_to_btds

def make_data_split(train_sz: float=80, test_sz: int=20, data_folder: str="./change-my-view-modes/v2.0/", save_folder: str="", shuffle: bool=True):
    
    if not os.path.isdir("./emnlp2017-bilstm-cnn-crf") and save_folder=="":
            raise ValueError("Can't find emnlp2017-bilstm-cnn-crf directory, \
                please specify save_folder in make_data_split()")
    
    if not os.path.isdir(data_folder):
        print("Can't find data folder:", data_folder, ".Downloading instead.")
        subprocess.call(shlex.split(f"git clone https://github.com/chridey/change-my-view-modes {shlex.quote(str(Path(data_folder).parent))}"))
    
    parser = make_data.get_parser()
    command = f"--train_sz {train_sz} --test_sz {test_sz} --data_folder {data_folder}"
    if shuffle:
        command += " --shuffle"
    args = parser.parse_args(shlex.split(command))
    out_folders = make_data.main(args)

    parser = xml_to_conll.get_parser()
    out_locs = {}
    for split, folder in out_folders.items():
        write_loc = os.path.join(str(Path(folder).parent), split+".txt")
        command = f"--folder {shlex.quote(folder)} --post_wise --write_file {shlex.quote(write_loc)}"
        args = parser.parse_args(shlex.split(command))
        xml_to_conll.main(args)
        out_locs[split] = write_loc
    
    if save_folder=="":
        save_folder = "./emnlp2017-bilstm-cnn-crf/data/cmv_modes"
    
    parser = conll_to_btds.get_parser()
    for split, out_file in out_locs.items():
        if split=="valid":
            split="dev"
        write_file = os.path.join(save_folder, split+".txt")
        command = f"--read_file {shlex.quote(out_file)} --write_file {shlex.quote(write_file)}"
        args = parser.parse_args(shlex.split(command))
        conll_to_btds.main(args)
    
    copy1, copy2 = save_folder.strip('/')+'1', save_folder.strip('/')+'2'
    
    if os.path.isdir(copy1):
        shutil.rmtree(copy1)
    if os.path.isdir(copy2):
        shutil.rmtree(copy2)
    
    subprocess.call(shlex.split(f"cp -r {save_folder} {copy1}"))
    subprocess.call(shlex.split(f"cp -r {save_folder} {copy2}"))
    
    return None

def read_naacl_dataset(dataset):
    naacl_data_dir = "./naacl18-multitask_argument_mining/dataSplits/fullData"
    all_data = [[]]
    for split in ["train", "dev", "test"]:
        with open(os.path.join(naacl_data_dir, dataset, split+".txt"), "r") as f:
            file_lines = f.readlines()
            for i, line in enumerate(file_lines):
                if line.strip()=="" and i!=0:
                    all_data[-1].append(line)
                    if i!=len(file_lines)-1:
                        all_data.append([])
                else:
                    all_data[-1].append(line)
        if not all_data[-1][-1].endswith("\n"):
            all_data[-1][-1] += "\n"
        
    print("Read", len(all_data), "sentences from:", dataset, "dataset")
    return all_data

def make_naacl_data_splits(train_sz, shuffle):
    if not os.path.isdir("./naacl18-multitask_argument_mining/"):
        raise ValueError("Can't find naacl18-multitask_argument_mining directory.")
    
    for dataset in ["essays", "hotel", "news", "var", "web", "wiki"]:
        
        all_data = read_naacl_dataset(dataset)
        if shuffle:
            random.shuffle(all_data)
        data_dir = os.path.join("./emnlp2017-bilstm-cnn-crf/data/", dataset)
        
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        
        os.makedirs(data_dir)
        
        train_data = all_data[:(train_sz*len(all_data))//100]
        test_data = all_data[(train_sz*len(all_data))//100:]
        
        with open(os.path.join(data_dir, "train.txt"), "w") as f:
            f.writelines(train_data)
        
        with open(os.path.join(data_dir, "test.txt"), "w") as f:
            f.writelines(test_data)
        
        with open(os.path.join(data_dir, "dev.txt"), "w") as f:
            f.writelines(all_data[-1])
    return None

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sz", type=float, default=80)
    parser.add_argument("--test_sz", type=int, default=20)
    parser.add_argument("--data_folder", type=str, default="./change-my-view-modes/v2.0/")
    parser.add_argument("--save_folder", type=str, default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--multi_data", action="store_true")
    args = parser.parse_args()
    
    assert args.train_sz+args.test_sz==100, "train_sz+test_sz must be 100"
    
    make_data_split(args.train_sz, args.test_sz, args.data_folder, args.save_folder, args.shuffle)
    if args.multi_data:
        make_naacl_data_splits(args.train_sz, args.shuffle)