"""
Script to convert CoNLL format data into the (b,t,d,s) format specified in 
the paper https://www.aclweb.org/anthology/P17-1002/ .

The third column of the CoNLL data has elements of form " [BIO_tag]-[C|P]:[n]:[relation_type] ", where
n is the first token position of the claim/premise the current claim/premise relates to. Example : "B-C:128:support"

For carrying out the MTL experiments, in the above paper, we add 2 more columns, splitting the third column
into 2 new columns, as follows:
First:  B-[C/P]
Second: [d]:[rel_type]

where d is the distance of the current component from the component it relates to (in terms of number of componenets).

To convert to the format of https://www.aclweb.org/anthology/N18-2006/ , 
as here : https://github.com/UKPLab/naacl18-multitask_argument_mining/blob/master/dataSplits/fullData/
use the script with the additional flags --remove_line_no and --remove_relations
"""

from typing import List, Tuple
from collections import Counter
import argparse, os

def get_components(str_lis: List[str]) -> List[int]:
    """
    str_lis: List corresponding to a paritcular paragraph/essay in the CoNLL format data

    Returns: A list containing component number of each element in str_lis.
    """
    components = list()
    cur_type = None
    component_no = -1

    for elem in str_lis:
        elem = elem.strip()
        pos, token, rel_entry = elem.split()
        first_extra_col = rel_entry.split(':')[0]
        
        try:
            component_type = first_extra_col.split('-')[1]
        except:
            component_type = 'O'
        
        if cur_type!=component_type:
            component_no+=1
            cur_type=component_type        
        
        components.append(component_no)
    
    return components

def add_cols(str_lis : List[str], components: List[Tuple[int, str]], args) -> List[str]:
    """
    str_lis: List corresponding to a paritcular paragraph/essay in the CoNLL format data
    components: A list containing component number of each element in str_lis.

    Returns: A new string list with added columns.
    """
    new_str_lis = []
    
    for elem in str_lis:
        elem = elem.strip()
        pos, token, rel_entry = elem.split()
        rel_entry_lis = rel_entry.split(':')
        first_extra_col = rel_entry_lis[0]
        
        if len(rel_entry_lis)>1 and rel_entry_lis[1]!='':
            rel_pos, rel_type = rel_entry_lis[1], rel_entry_lis[2]
            d = str( components[int(rel_pos)-1] - components[int(pos)-1] )
        else :
            rel_type = str(None)
            d = str(None)
        
        rel_entry = first_extra_col if rel_entry.startswith('O') else first_extra_col+':'+d+':'+rel_type
        
        elem = ( ('' if args.remove_line_no else pos + '\t') +
                  token + '\t' + ('' if args.remove_relations else rel_entry + '\t')+
                  first_extra_col+('' if args.remove_relations else '\t'+d+':'+rel_type)+'\n'
        )
        
        new_str_lis.append(elem)

    return new_str_lis

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_file', type=str, help='The file having the CoNLL format data.')
    parser.add_argument('--write_file', type=str, help='The location where the script should write the new file with added columns.')
    parser.add_argument('--remove_line_no', action='store_true', help='If this flag is provided, the line numbers are remmoved.')
    parser.add_argument('--remove_relations', action='store_true', help='If this flag is provided, the relation distance and ')
    return parser

def main(args):
    added_cols_strs = []
    essay_lis = []
    with open(args.read_file, 'r') as f:
        old_strs = f.readlines()
        for i, line in enumerate(old_strs):
            
            if line.strip()=='' or i==len(old_strs):
                added_cols_strs.append('\n')
                components = get_components(essay_lis)
                essay_lis = add_cols(essay_lis, components, args)
                added_cols_strs += essay_lis
                essay_lis = []
            
            else:
                essay_lis.append(line)
    
    os.makedirs(os.path.dirname(args.write_file), exist_ok=True)
    with open(args.write_file, 'w') as f:
        for elem in added_cols_strs:
            f.write(elem)
    
if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)