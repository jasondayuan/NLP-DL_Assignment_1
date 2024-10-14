import time
import random
import numpy as np
import itertools
from collections import Counter
import pdb

def flatten_list(nested_list: list, mode=0):
    if mode == 0:
        result_list = []
        for lst in nested_list:
            result_list.extend(lst)
        return result_list
    else:
        return list(itertools.chain(*nested_list))


def char_count(s: str, mode=0):
    if mode == 0:
        result_dict = {}
        for char in s:
            if char in result_dict:
                result_dict[char] += 1
            else:
                result_dict[char] = 1
        return result_dict
    else:
        return dict(Counter(s))

def flatten_list_prof(scale, mode):
    # Generate sample
    sample = []

    left = scale
    while left > 0:
        lst_len = np.random.geometric(0.005) - 1
        if lst_len < left:
            sample.append([random.randint(0, 100) for _ in range(lst_len)])
            left -= lst_len
        elif lst_len > left:
            sample.append([random.randint(0, 100) for _ in range(left)])
            break
        else:
            sample.append([random.randint(0, 100) for _ in range(lst_len)])
            break
        
    # Testing
    start_time = time.perf_counter()
    result = flatten_list(sample, mode)
    timespan = time.perf_counter() - start_time
    assert len(result) == scale # sanity check

    return timespan

def char_count_prof(scale, mode):
    common_characters = [chr(i) for i in range(32, 127)]

    # Generate sample
    sample = ""
    for _ in range(scale):
        sample += random.choice(common_characters)
        
    # Testing
    start_time = time.perf_counter()
    result = char_count(sample, mode)
    timespan = time.perf_counter() - start_time
    # sanity check
    tot_len = 0
    for _, value in result.items():
        tot_len += value
    assert tot_len == scale

    return timespan

if __name__ == "__main__":
    scales = [int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]
    TESTS_PER_SCALE = 10
    
    # flatten_list
    test_results = {scale:[] for scale in scales}

    for _ in range(TESTS_PER_SCALE):
        for scale in scales:
            test_results[scale].append(flatten_list_prof(scale, 0))

    for key, value in test_results.items():
        print(f"flatten_list 0 Scale: {key} AvgTime: {np.mean(np.array(value))*1000:.4f}")

    test_results = {scale:[] for scale in scales}

    for _ in range(TESTS_PER_SCALE):
        for scale in scales:
            test_results[scale].append(flatten_list_prof(scale, 1))

    for key, value in test_results.items():
        print(f"flatten_list 1 Scale: {key} AvgTime: {np.mean(np.array(value))*1000:.4f}")
    

    # char_count
    test_results = {scale:[] for scale in scales}
    
    for _ in range(TESTS_PER_SCALE):
        for scale in scales:
            test_results[scale].append(char_count_prof(scale, 0))

    for key, value in test_results.items():
        print(f"char_count 0 Scale: {key} AvgTime: {np.mean(np.array(value))*1000:.4f}")

        test_results = {scale:[] for scale in scales}
    
    for _ in range(TESTS_PER_SCALE):
        for scale in scales:
            test_results[scale].append(char_count_prof(scale, 1))

    for key, value in test_results.items():
        print(f"char_count 1 Scale: {key} AvgTime: {np.mean(np.array(value))*1000:.4f}")