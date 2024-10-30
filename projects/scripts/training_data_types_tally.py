import json 
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="given a VGG style annotation json, count the number of each type of object in the annotation")
parser.add_argument("-s", "--source", required=True, help="the json annot")
parser.add_argument("-l", "--labels", required=True, help="list of labels which the json contains")


args = parser.parse_args()

LABELS = []
with open(os.path.normpath(args.labels), "r") as file:
    labels = file.read()  
    LABELS = labels.strip().split('\n')


annotation = {}
with open(os.path.normpath(args.source)) as file:
    annotation = json.load(file)



tallies = np.zeros(len(LABELS)).astype(int)

for value in annotation.values():
    regions = value["regions"]
    for mask in regions.values():
        label = mask["region_attributes"]["label"]
        index = LABELS.index(label)
        tallies[index] += 1

sum = np.sum(tallies, axis=0)

for i in range(len(tallies)):
    print(f"{LABELS[i]:<20} : {tallies[i]:<5} || 比例: {tallies[i]/sum:<5.2f}")