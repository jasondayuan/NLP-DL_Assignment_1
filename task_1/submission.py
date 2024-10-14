import itertools
from collections import Counter

def flatten_list(nested_list: list):
    result_list = []
    for lst in nested_list:
        result_list.extend(lst)
    return result_list


def char_count(s: str):
    result_dict = {}
    for char in s:
        if char in result_dict:
            result_dict[char] += 1
        else:
            result_dict[char] = 1
    return result_dict