#!/usr/bin/python

from __future__ import print_function
from os import mkdir, walk
from os.path import join, exists
from copy import copy
from shutil import copyfile
from functools import reduce
import operator
import sys
import re

# Source directory (raw)
SRC_DIR = "./raw/"

# Major class tags
posture_tags = ["sitting", "standing"]
action_tags = ["up", "down"]

# Associated tags (e.g. sitting involves a chair in our experiments)
associated_tags = {"sitting": ["chair"], "standing": ["chair"]}

# Write some regular expressions that match folder names for extra tags
extra_tags = {'table' : [".*masa.*"]}

def prepare_posture_dataset():
    print("Preparing posture dataset")
    mkdir("datasets/posture")

    dataset_info = {}
    for tag in posture_tags:
        # Compute the full list of default tags
        default_tags = [tag]
        if tag in associated_tags:
            default_tags.extend(associated_tags[tag])

        # Find the folders that match the major class (tag)
        tag_filter = lambda dirname: dirname.upper().endswith("/" + tag.upper())
        for exp in [r for (r, _, _) in walk(SRC_DIR) if tag_filter(r)]:
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
                    dataset_info[name] = (join(r,f), copy(folder_tags))

    # Write all to file and copy images
    with open("datasets/posture/info", "w") as out:
        for (f, info) in dataset_info.items():
            copyfile(info[0], join("datasets/posture/", f))
            out.write(f + "," + ",".join(info[1]) + "\n")

    print("Done")

def choose_action_tag(tag1, tag2):
    bad_pairs = [["down", "up"], ["sitting", "standing"]]
    if tag1 != tag2 and sorted([tag1, tag2]) in bad_pairs:
        print("This should not happen: %s vs %s!!!" % (tag1, tag2))
        return None
    elif "down" in [tag1, tag2]:
        return "down"
    elif "up" in [tag1, tag2]:
        return "up"
    else:
        raise Exception("Unhandled behavior")

def prepare_action_dataset():
    print("Preparing action dataset")
    mkdir("datasets/action")

    idx = 0
    count = 0
    conc = lambda x, y: list(x) + list(y)
    different_tags = {}
    # Take each folder that contains a direct subfolder with an action name
    is_parent_dir = lambda ds: any([tag in ds for tag in action_tags])
    for d in [r for (r,ds,_) in walk(SRC_DIR) if is_parent_dir(ds)]:
        # Create experiment directory
        root_dir = "datasets/action/%03d/" % idx
        idx = idx + 1
        mkdir(root_dir)
        # Go through all files
        sets = [list(map(lambda x: join(r2, x), fs)) for (r2,_,fs) in walk(d)]
        files = {}
        for f in reduce(conc, sets):
            count += 1
            jpg = f.split("/")[-1]
            crt_tag = f.split("/")[-3]
            different_tags[crt_tag] = 1
            if jpg in files:
                # Same file occurs with two tags
                winner_tag = choose_action_tag(crt_tag, files[jpg]["tag"])
                if winner_tag == crt_tag:
                    files[jpg] = {"path": f, "tag": crt_tag}
            else:
                files[jpg] = {"path": f, "tag": crt_tag}
        # Copy files
        with open(join(root_dir, "info"), "w") as out:
            for (jpg,info) in sorted(files.items(), key=operator.itemgetter(0)):
                copyfile(info["path"], join(root_dir, jpg))
                out.write("%s,%s\n" % (jpg, info["tag"]))
    print(different_tags)
    print("Copied all %d files!" % count)
    print("Done")

if __name__ == "__main__":
    print("Preparing data sets...")
    # Check if folder already exists
    if exists("datasets"):
        print("[ERROR] Folder 'datasets' exists! Solve this first (by deleting it)!")
        exit()
    mkdir("datasets")
    # Create datasets
    prepare_posture_dataset()
    prepare_action_dataset()
    print("Done all!")
