#!/usr/bin/python

from __future__ import print_function
from os import mkdir, walk
from copy import copy
from shutil import copyfile
import sys
import os.path
import re

# Major class tags
posture_tags = ["sitting", "standing"]
action_tags = ["up", "down"]

# Associated tags (e.g. sitting involves a chair in our experiments)
associated_tags = {"sitting": ["chair"], "standing": ["chair"]}

# Write some regular expressions that match folder names for extra tags
extra_tags = {
    'table' : [".*masa.*"]
}

def prepare_posture_dataset():
    print("Preparing posture dataset")
    mkdir("datasets/posture")

    src_dir = sys.argv[1]
    dataset_info = {}

    for tag in posture_tags:
        # Compute the full list of default tags
        default_tags = [tag]
        if tag in associated_tags:
            default_tags.extend(associated_tags[tag])

        # Find the folders that match the major class (tag)
        tag_filter = lambda dirname : dirname.upper().endswith("/" + tag.upper())
        for exp in [r for (r, _, _) in walk(src_dir) if tag_filter(r)]:
            for (r, fs) in [(r, fs) for (r, _, fs) in walk(exp) if fs]:
                # Compute all tags for current folder
                folder_tags = copy(default_tags)
                for (et, regexs) in extra_tags.items():
                    for regex in regexs:
                        if any([re.match(regex, d) for d in r.split("/")]):
                            folder_tags.append(et)

                # Go through all files in folder
                for f in fs:
                    name = f
                    while name in dataset_info:
                        name = name[:-4] + "_" + name[-4:]
                    dataset_info[name] = (os.path.join(r,f), copy(folder_tags))

    # Write all to file and copy images
    with open("datasets/posture/info", "w") as out:
        for (f, info) in dataset_info.items():
            copyfile(info[0], os.path.join("datasets/posture/", f))
            out.write(f + "," + ",".join(info[1]) + "\n")

    print("Done")

def prepare_action_dataset():
    print("Preparing action dataset")
    print("Done")
    pass


if __name__ == "__main__":
    print("Preparing data sets...")
    # Check if folder already exists
    if os.path.exists("datasets"):
        print("Folder 'datasets' exists! Solve this first (by deleting it)!")
        exit()
    os.mkdir("datasets")
    # Create datasets
    prepare_posture_dataset()
    prepare_action_dataset()
    print("Done all!")
