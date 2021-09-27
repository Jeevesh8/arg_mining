import re
from typing import List, Tuple

def refine(annotations: List[str], keep_data_type: bool=True) -> List[Tuple[str, int, int]]:
    """
    Args:
        annotations:    The annotations read from a .ann file using readlines()

        keep_data_type: Whether to keep annotations of type "data". If False,
                        only "background_claim" and "own_claim" type annotations
                        will be in the final list.
    Returns:
        A list of tuples of form (component type, start index, end index), where:
            1. component type is one of ["background_claim", "own_claim", "data"]
            2. start_idx and end_idx are the indices denoting the span of characters
               corresponding to the component in the .txt file.
        
        And a list of tuples of form (relation type, arg1 id, arg2 id), where:
            1. relation type is one of ["supports", "contradicts", "parts_of_same"]
            2. arg1 id and arg2 id are the ids of the components as specified in .ann 
               files.
    
    NOTE: The returned list of component types is sorted in th order of occurance of spans 
          in the  .txt file(i.e., in ascending order of start index)
    """
    refined_annotations = []
    rel_type_annotations = []

    for i, annotation in enumerate(annotations):
        id, info = annotation.split('\t')[0:2]
        
        if id[0]=='R':
            rel_type, comp1, comp2 = info.split()
            
            if rel_type=="semantically_same":
                rel_type = "parts_of_same"
            
            if rel_type not in ["supports", "contradicts", "parts_of_same"]:
                raise ValueError("Relation type not recognized:", rel_type)
            
            comp1, comp2 = comp1.split(":")[1], comp2.split(":")[1]
            rel_type_annotations.append((rel_type, comp1, comp2))
            
            continue
        
        if not keep_data_type and 'claim' not in info.split(' ')[0].split('_'):
            continue
        
        if id[0]!='T':
            raise ValueError("Unknown annotation type: "+id[0])
        
        try:
            component_type, start_idx, end_idx = info.split(' ')
        except ValueError as e:
            if str(e)=="too many values to unpack (expected 3)":
                try:
                    component_type, partitions = info.split(' ', 1)
                    start_idx, end_idx = partitions.split(' ')[0], partitions.split(' ')[-1]
                except:
                    raise ValueError("Encountered unknown annotation span type:", 
                                     info.split())
            else:
                raise e
            
        if component_type not in ["background_claim", "own_claim", "data"]:
            
            raise ValueError("Unknown component type: " + 
                             component_type +" in annotation number: "+str(i+1))
        
        refined_annotations.append((id, component_type, int(start_idx), int(end_idx)))
        
    return sorted(refined_annotations, key=lambda annotation: annotation[2]), rel_type_annotations

def add_component_type_tags(paper_str: str, 
                            refined_annotations: List[Tuple[str, int, int]]) -> str:
    """Adds <comp_type id="Tn">.*</comp_type> tags around the text in 
    paper_str corresponding to the components present in 
    refined_annotations.
    """
    new_paper_str = ""
    zero_idx = 0
    for annotation in refined_annotations:
        id, comp_type, start_idx, end_idx = annotation
        start_idx, end_idx = start_idx-zero_idx, end_idx-zero_idx
        #print(start_idx, end_idx)
        whitespace_start_pos = re.search(r"(\s*)$",
                                        paper_str[:start_idx]).start(0)
        #print("whitespace start:", whitespace_start_pos)
        new_paper_str += paper_str[:whitespace_start_pos]
        new_paper_str += "<"+comp_type+" "+"id=\""+id+"\">"
        new_paper_str += paper_str[whitespace_start_pos:end_idx]
        new_paper_str += "</"+comp_type+">"

        paper_str = paper_str[end_idx:]
        zero_idx += end_idx
    return new_paper_str
