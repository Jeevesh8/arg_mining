import re
from typing import List, Tuple

def merge_subsections(heading_sections: List[Tuple[str, str]], 
                      joiner: str) -> List[Tuple[str, str]]:
    """Merges together subsections with headings which have numbers with the same 
    integer parts in them.
    
    Preconditions:
        1. Assumes only heading tags, and their subsections are there in heading_sections.
           No (title, abstract) pairs etc.
    
    Args:
        heading_sections: List of tuples of form (heading, section content)
        joiner:           String to use when joining together headings and content
                          of subsection to the contect of parent section.
    Returns:
        List of tuples of form (heading, content); where each tuple corresponds to 
        a highest level section of the paper.
    """
    merged_h_n_s = []
    prev_section_num = str(0)
    
    for heading, section in heading_sections:
        merged_h_n_s.append((heading, section))
        
        try:
            section_num = re.search(r"\s*(\d+).+", heading).group(1)
        except AttributeError:
            prev_section_num = str(0)
            continue
        
        if section_num==prev_section_num and len(merged_h_n_s)>=2:
            merged_h_n_s[-2] = (merged_h_n_s[-2][0], (merged_h_n_s[-2][1]+joiner
                                                      +merged_h_n_s[-1][0]+joiner
                                                      +merged_h_n_s[-1][1]))
            merged_h_n_s.pop()
            
        prev_section_num = section_num
    
    return merged_h_n_s

def break_into_sections(paper_str: str, merge_subsecs: bool=True, 
                        joiner: str="[NEWLINE]") -> List[Tuple[str, str]]:
    """Breaks the paper into sections. 
    Preconditions:
        1. Paper is expected to contain sections/sub-sections marked by only <Title>, <Abstract>, 
           and heading tags, lying directly under the <Document> tag.
    
        2. Subsections are assumed to be indicated by a number with decimal in front 
           of the heading's name. And may be merged into the parent section, by 
           application of ______ function to the output of this function.
        
        3. References section is assumed to equivalent to any section that has
           "references" or "bibliography" (case-insensitive) in its heading. 
           These sections are omitted from the returned list.
    
    Args:
       paper_str:      The string read from the file, with added claims/data tags.

       merge_subsecs:  If True, the subsections denoted by decimal point are 
                       joined togther into a single subbsection with heading of 
                       the parent section.
       
       joiner:         The string to add between two merged subsections, if any.

    Returns:
        A list of tuples of form (heading, content).
    
    Note: Title and Abstract sections in paper_str are always merged into one section. 
    Title is treated as heading for abstract.
    """
    try:
        title  = re.search(r"(?s)<Title>(.+)</Title>", paper_str).group(1)
    except AttributeError as e:
        raise ValueError("Can't find title in provided paper:", paper_str)
    
    try:    
        abstract = re.search(r"(?s)<Abstract>(.+)</Abstract>", paper_str).group(1)
    except AttributeError as e:
        raise ValueError("Can't find Abstract in provided paper:", paper_str)
    
    heading_sections = [elem[::-1] 
                        for elem in re.findall(r"(?s).+?>\d+H/<.+?>\d+H<", 
                                               paper_str[::-1])][::-1]
    
    heading_sections = [(re.search(r"(?s)<H\d+>(.+)</H\d+>(.+)", section).group(1),
                         re.search(r"(?s)<H\d+>(.+)</H\d+>(.+)", section).group(2))
                        for section in heading_sections]
    
    heading_sections = list(filter(lambda h_n_s : (h_n_s[0].lower().find('references')==-1 and 
                                                   h_n_s[0].lower().find('bibliography')==-1), 
                                   heading_sections)
                           )
    if merge_subsecs:
        heading_sections = merge_subsections(heading_sections, joiner)  
        
    return [(title, abstract)]+heading_sections
