from typing import Optional

import numpy as np

from .configs import config

def get_transition_mat(
        min_val: Optional[float] = -1.0,
        max_val: Optional[float] = 1.0,
        pt_transition_mat: Optional[np.ndarray] = None,
        begin_to_begin: Optional[bool] = False) -> np.ndarray:
    """
    Args:
        min_val: minimum value to use for any transition in the matrix
        max_val: maximum value to use for any transition in the matrix
        
        Uses a pre-trained transition matrix(if provided) of form:
               B-P  I-P
        B-P [[  x ,  y ],
        I-P [[  z ,  w ]]
        (where 'P' stands for post) to initialize a new transition matrix for
        argument mining fintuning. The transition matrix defines transitions
        amongst ['B-C', 'I-C', 'B-P', 'I-P', 'O'].

        begin_to_begin: whether to consider transitions from {'B-C', 'B-P'} to themselves. If True,
                        then the corresponding entries are initalised with 1e-3, otherwise with -np.inf.
        
    Returns:
        transition_matrix of shape [5, 5] with values uniform values in range [min_val, max_val]; 
        (i, j)-th entry is the transition energy of transitioning from state i to state j.
    """
    
    rng = np.random.default_rng(12345)

    random_transition_mat = (rng.uniform(low=min_val, high=max_val, size=(5, 5),))
    ac_dict = config["arg_components"]

    random_transition_mat[[ac_dict["I-C"], ac_dict["other"]],
                          ac_dict["I-P"]] = -np.inf
    random_transition_mat[[ac_dict["I-P"], ac_dict["other"]],
                          ac_dict["I-C"]] = -np.inf
    random_transition_mat[ac_dict["other"],
                          [ac_dict["I-P"], ac_dict["I-C"]]] = -np.inf

    random_transition_mat[ac_dict["B-C"], ac_dict["I-P"]] = -np.inf
    random_transition_mat[ac_dict["B-P"], ac_dict["I-C"]] = -np.inf

    if not begin_to_begin:
        random_transition_mat[[ac_dict["B-C"], ac_dict["B-P"]], 
                              [ac_dict["B-C"], ac_dict["B-P"]]] = -np.inf
    else:
        random_transition_mat[[ac_dict["B-C"], ac_dict["B-P"]], 
                              [ac_dict["B-C"], ac_dict["B-P"]]] = 1e-3
    
    if pt_transition_mat is not None:
        random_transition_mat[
            [ac_dict["B-C"], ac_dict["B-P"]],
            [ac_dict["B-C"], ac_dict["B-P"]]] = pt_transition_mat[0, 0]

        random_transition_mat[ac_dict["B-C"],
                            ac_dict["I-C"]] = pt_transition_mat[0, 1]
        random_transition_mat[ac_dict["B-P"],
                            ac_dict["I-P"]] = pt_transition_mat[0, 1]

        random_transition_mat[
            [ac_dict["I-P"], ac_dict["I-C"]],
            [ac_dict["B-P"], ac_dict["B-C"]]] = pt_transition_mat[1, 0]

        random_transition_mat[ac_dict["I-C"],
                            ac_dict["I-C"]] = pt_transition_mat[1, 1]
        random_transition_mat[ac_dict["I-P"],
                            ac_dict["I-P"]] = pt_transition_mat[1, 1]

    print("Initialized Transition matrix(from-to):")
    print(random_transition_mat)

    return np.transpose(np.array(random_transition_mat))
