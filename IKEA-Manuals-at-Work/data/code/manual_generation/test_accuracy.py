import ast
import copy
from permutations import print_trees
from manual_generation.models import Model, SinglePartModel, SimilarityModel
from utils.meters import Meters
from manual_generation.eval import eval_assembly_tree
from tqdm import tqdm
from pprint import pprint
import argparse
from manual_generation.dataset import Dataset
from utils.data import build_tree_from_list, tree_to_list
import json
import sys
# sys.setrecursionlimit(30) 

check = True
num = 5

def evaluate(model: Model, dataset: Dataset, check_symm: bool):
    meters = Meters()
    count = 0
    improved_count = 0
    totally_correct_trees = 0
    for f in tqdm(dataset):
        found_iter = False
        if f['part_ct']<=num:
        # if f['part_ct']>=7:
        # if f['name'] == 'glenn':
            count += 1
            tree_gt = f['tree']
            print(f"{f['category']}/{f['name']}")
            #with open(f"../../out_trees_v9_revert/{f['category']}/{f['name']}/tree.json", "r") as file:
            with open(f"../../out_trees_v14_no_seg/{f['category']}/{f['name']}/tree.json", "r") as file:
                tree_str = json.load(file)
            tree_list = json.loads(tree_str)
            # print(tree_list)
            # print(tree_list, tree_store)
            # print(tree_list, type(tree_list))
            tree_pred = build_tree_from_list(tree_list)
            if not check_symm:
                print("USING ORIGINAL MODEL")
                tree_pred = model(f)
                tree_list = tree_to_list(tree_pred)
                # print(tree_list)
            tree_gt_list = tree_to_list(tree_gt)
            if are_nested_lists_equal(tree_list, tree_gt_list):
                totally_correct_trees += 1
                print(f"NORMAL Predicted: {tree_list}, GT: {tree_gt_list}")
                found_iter = True
            else:
                with open(f"../../output/{f['category']}/{f['name']}/equiv_parts.txt", "r") as text_file:
                    data = text_file.read()
                nested_list = ast.literal_eval(data)
                tree_store = []
                # print(tree_list, nested_list)
                print_trees(copy.deepcopy(tree_list), nested_list, tree_store)
                for indiv_tree in tree_store:
                    if f['part_ct']>=11:
                        break
                    if are_nested_lists_equal(indiv_tree, tree_gt_list):
                        totally_correct_trees += 1
                        improved_count += 1
                        print(f"IMPROVED Predicted: {indiv_tree}, GT: {tree_gt_list}")
                        found_iter = True
                        break
                if not found_iter:
                    print(f"NO TREES FOUND: Predicted: {indiv_tree}, GT: {tree_gt_list}")

    print(f"{num} parts count: {count}")
    print(improved_count)
    print(f"final score: {totally_correct_trees}")
    return meters

def find_max_val(dict):
    # Find largest value in result_tmp dictionary
    largest_value = float('-inf')
    for key, subdict in dict.items():
        if key != "no_children":
            for metric, value in subdict.items():
                # Update the largest value if the current value is greater
                if value > largest_value:
                    largest_value = value
    return largest_value


def are_nested_lists_equal(list1, list2):
    # Create deep copies of the lists to avoid modifying the originals
    list1_copy = copy.deepcopy(list1)
    list2_copy = copy.deepcopy(list2)
    
    # If both are not lists, compare directly
    if not isinstance(list1_copy, list) or not isinstance(list2_copy, list):
        return list1_copy == list2_copy
    
    # If lengths are different, they can't be equal
    if len(list1_copy) != len(list2_copy):
        return False
    
    # Recursively check each element in the lists
    for item1 in list1_copy:
        found_match = False
        for item2 in list2_copy:
            if are_nested_lists_equal(item1, item2):
                found_match = True
                list2_copy.remove(item2)  # Remove the matched item to avoid re-matching
                break
        if not found_match:
            return False
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_features_pkl', type=str)
    parser.add_argument('--data_json', type=str)
    parser.add_argument('--parts_dir', type=str)
    args = parser.parse_args()

    dataset = Dataset(data_json=args.data_json, parts_dir=args.parts_dir, part_features_pkl=args.part_features_pkl)
    evaluate_models = ['single_part', 'similarity']
    # evaluate_models = ['single_part']
    # evaluate_models = ['similarity']
    if 'single_part' in evaluate_models:
        single_part_model = SinglePartModel()
        meters_single = evaluate(single_part_model, dataset, check)
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')

    if 'similarity' in evaluate_models:
        similarity_model = SimilarityModel()
        meters = evaluate(similarity_model, dataset, check)
    #     pprint('Similarity Model:')
    #     pprint(meters.avg_dict())
    #     # for k, v in meters.avg_dict().items():
    #     #     print(k, end=' ')
    # pprint('Single Part Model:')
    # pprint(meters_single.avg_dict())
