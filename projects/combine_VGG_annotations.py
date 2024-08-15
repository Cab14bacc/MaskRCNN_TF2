import argparse 
import json
import os
import glob

parser = argparse.ArgumentParser(description="Take in VGG annotations in different JSON files in the source directory, and combine them into one JSON file")
parser.add_argument("-o", "--output", default="annot.json", help="output json filepath and name")
parser.add_argument("-s", "--source", default="./annot", help="source directory")

args = parser.parse_args()

# result json
result = {}

# for all files in source, load and add to result
for filename in glob.glob(os.path.join(args.source, "*.json")):
    with open(os.path.join(os.getcwd(), filename)) as f:
        result.update(json.load(f))

print(result.keys())
result = dict(sorted(result.items(), key = lambda item : item[0]))

# output result to result json
with open(os.path.normpath(args.output), "w") as outputFile:
    json.dump(result, outputFile)

