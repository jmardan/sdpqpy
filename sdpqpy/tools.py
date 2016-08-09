# -*- coding: utf-8 -*-
"""
Tools used by the main classes.

@author: Christian Gogolin, Peter Wittek
"""
import csv


def unique_permutations(elements):
    """Taken from
    http://stackoverflow.com/questions/6284396/permutations-with-unique-values
    """
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation


def write_array(filename, array):
    with open(filename, 'w') as file_:
        writer = csv.writer(file_)
        writer.writerow(array)
