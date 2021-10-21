# Script to convert CMV Modes xml data into CoNLL format

import os, re
import bs4
from bs4 import BeautifulSoup
import argparse

def clean_text(text: str) -> str:
    replaces = [("’", "'"), ("“", '"'), ("”", '"')]
    for elem in replaces:
        text = text.replace(*elem)
    
    for elem in ['.', ',','!',';', ':', '*', '?', '/', '\'', '\"', '[', ']', '(', ')', '_', '^', '>']:
        text = text.replace(elem, ' '+elem+' ')
    
    text = text.strip(' _\t\n')
    text = text.split('____')[0]                                                    #To remove footnotes
    text = text.strip(' _\t\n')
    text = re.sub(r'\(https?://\S+\)', '<url>', text)                               #To remove URLs
    text = re.sub(r'&gt;.*(?!(\n+))$', '', text)                                    #To remove quotes at last.
    text = text.replace("&", "and")
    text = text.rstrip(' _\n\t')
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = text.lower()
    return text

def make_ref_dic(parsed_xml: bs4.BeautifulSoup) -> dict:
    """
    parsed_xml: The parsed xml of the entire thread

    Returns: dictionary with keys as ids of claims/premises and values as the line number they will start on.
    """
    
    ref_dic = {}
    title_text = str(parsed_xml.find('title').find('claim').contents[0])
    ref_dic['title'] = 1
    n_lines = len(clean_text(title_text).split())+1

    for post in [parsed_xml.find('op')]+parsed_xml.find_all('reply'):
        
        for elem in post.contents:
        
            elem = str(elem)
            
            if elem.startswith('<claim'):
                parsed_claim = BeautifulSoup(elem, "xml")
                ref_dic[parsed_claim.find('claim')['id']] = n_lines
                n_lines += len(clean_text( str(parsed_claim.find('claim').contents[0]) ).split())
            
            elif elem.startswith('<premise'):
                parsed_premise = BeautifulSoup(elem, "xml")
                ref_dic[parsed_premise.find('premise')['id']] = n_lines
                n_lines += len(clean_text( str(parsed_premise.find('premise').contents[0]) ).split())
            
            else:
                n_lines += len(clean_text(elem).split())
    return ref_dic

def add_cp(cp_text: str, ref_dic: dict, id: str, ref: str=None, rel :str=None, type='C'):
    """
    Appends tokenwise labelled text to str_to_write for a particular claim/premise.
    
    ref_dic: dictionary with keys as ids of claims/premises and values as the line number they will start on.
    
    id:   id of the current claim/premise as provided in AMPERSAND data, 
          used by other claims/premise to support this comment
    
    ref: The id of the claim/premise the current claim/premise depends on.

    rel: The type of relation of the current claim/premise with the claim/premise id in ref('support', 'agreement',...)

    type: 'C' if cp_text corresponds to claim; 'P' if cp_text corresponds a premise
    """
    global str_to_write

    i = int(str_to_write[-1].split('\t')[0])+1 if id!='title' and str_to_write[-1]!='\n' else 1
    
    cp_text = clean_text(cp_text)
    cp_text = cp_text.split()
    
    for j, token in enumerate(cp_text):
        related_cp_start = str(ref_dic[ref])+':' if ref is not None and ref in ref_dic else ''
        relation_entry = (f'B-{type}:' if j==0 else f'I-{type}:')+str(related_cp_start)+(rel if rel is not None and related_cp_start != '' else '')
        str_to_write.append(str(i)+'\t'+token+'\t'+relation_entry)
        i+=1

def refine_ref_dic(post: bs4.element.Tag, ref_dic: dict, start_idx: int) -> dict:
    """
    Filters ref_dic to remove references to claim/premise outside of the post given. 
    If such references are not there, then add_cp will not include inter-turn relations.

    post: output of parsed_xml.find('op'), or an element of parsed_xml.find_all('reply')
          where 'parsed_xml' is the output of parsing the entire xml string of a .xml file in AMPERSAND
    
    ref_dic: dictionary with keys as ids of claims/premises and values as the line number they will start on.

    start_idx: Index in str_to_write, from where the posts of current thread start.

    Returns: the filtered dictionary
    """
    global str_to_write
    last_index = 0
    for j, elem in enumerate(str_to_write[start_idx:], start=start_idx):
        if j+1<len(str_to_write) and str_to_write[j+1]=='\n':
            last_index += int(str_to_write[j].split('\t')[0])

    new_ref_dic = {}
    if str(post).startswith('<op'):
        new_ref_dic['title'] = ref_dic['title']

    for elem in post.contents:
        elem = str(elem)

        if elem.startswith('<claim'):
            parsed_claim = BeautifulSoup(elem, "xml")
            new_ref_dic[parsed_claim.find('claim')['id']] = ref_dic[parsed_claim.find('claim')['id']]-last_index
        elif elem.startswith('<premise'):
            parsed_premise = BeautifulSoup(elem, "xml")
            new_ref_dic[parsed_premise.find('premise')['id']] = ref_dic[parsed_premise.find('premise')['id']]-last_index
    
    return new_ref_dic

def build_CoNLL(parsed_xml: bs4.BeautifulSoup, thread_wise: bool=True):
    """
    parsed_xml: the output of parsing the entire xml string of a .xml file in AMPERSAND
    thread_wise: boolean to choose whether to write entire thread as one unit 
                 with inter-comment relations or to write each post and comment separately.
    
    Writes the entire thread to str_to_write, in the CoNLL format.
    """
    global str_to_write
    
    if len(str_to_write)==0 or str_to_write[-1]!='\n':
        str_to_write.append('\n')
    
    start_idx = len(str_to_write)-1
    
    temp_ref_dic = make_ref_dic(parsed_xml)
    
    title_text = str(parsed_xml.find('title').find('claim').contents[0])
    add_cp(title_text, temp_ref_dic, id='title')
    
    for post in [parsed_xml.find('op')]+parsed_xml.find_all('reply'):
        
        if thread_wise:
            ref_dic = temp_ref_dic
        else:
            ref_dic = refine_ref_dic(post, temp_ref_dic, start_idx)
        
        for elem in post.contents:
        
            elem = str(elem)
            
            if elem.startswith('<claim'):
                parsed_claim = BeautifulSoup(elem, "xml")
                try:
                    add_cp(clean_text( str(parsed_claim.find('claim').contents[0]) ),
                           ref_dic, parsed_claim.find('claim')['id'], parsed_claim.find('claim')['ref'], 
                           parsed_claim.find('claim')['rel'])
                except:
                    add_cp(clean_text( str(parsed_claim.find('claim').contents[0]) ),
                           ref_dic, parsed_claim.find('claim')['id'], None, 
                           None)
                
            elif elem.startswith('<premise'):
                parsed_premise = BeautifulSoup(elem, "xml")
                try:
                    add_cp(clean_text( str(parsed_premise.find('premise').contents[0]) ),
                           ref_dic, parsed_premise.find('premise')['id'], parsed_premise.find('premise')['ref'], 
                           parsed_premise.find('premise')['rel'], 'P')
                except:
                    add_cp(clean_text( str(parsed_premise.find('premise').contents[0]) ),
                           ref_dic, parsed_premise.find('premise')['id'], None, 
                           None, 'P')
        
            else:
                i = int(str_to_write[-1].split('\t')[0])+1 if str_to_write[-1]!='\n' else 1
                for token in clean_text(elem).split():
                    str_to_write.append(str(i)+'\t'+token+'\t'+'O')
                    i+=1
        
        if not thread_wise:
            str_to_write.append('\n')

str_to_write = []

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='Folder having the .xml files of AMPERSAND data.')
    parser.add_argument('--post_wise', action='store_true', 
                        help='if this flag is provided, then the inter-comment relations will be ignored and data will be constructed in a post-by-post format.')
    parser.add_argument('--write_file', type=str, help='Filename where the script should write the data in CoNLL format.')
    return parser

def main(args):
    global str_to_write
    str_to_write = []
    for f in os.listdir(args.folder):
        filename = os.path.join(args.folder, f)
        if os.path.isfile(filename) and filename.endswith('.xml'):
            with open(filename, 'r') as g:
                xml_str = g.read()
            parsed_xml = BeautifulSoup(str(BeautifulSoup(xml_str, "lxml")), "xml")
            build_CoNLL(parsed_xml, not args.post_wise)
    
    os.makedirs(os.path.dirname(args.write_file), exist_ok=True)
    with open(args.write_file, 'w') as f:
        for elem in str_to_write[1:]:
            f.write(elem+'\n' if not elem.endswith('\n') else elem)

if __name__=='__main__':
    parser = get_parser()
    
    args = parser.parse_args()
    
    main(args)