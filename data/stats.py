#!/usr/bin/env python

'''
Script that analyses raw input folders. One or more folders can be given as
(a list of) command line arguments. The folders must contain only dataset
related files. The script only takes into account the fact that the parent
of each leaf folder (individiual sequence) represents the TAG (class) for that
sequence. Measures are computed per TAG.
Measures:
    * Number of sequences
    * Mean number of frames per sequence
    * Pop. std. dev. number of frames per sequence
    * Mean elapsed time per sequence
    * Pop. std. dev. elapsed time per sequence

For pretty printing in a table please install terminaltables:
    pip install terminaltables
'''

from __future__ import print_function
from os.path import join
from os.path import isfile
from os import walk
from os import listdir
from sys import argv
import numpy


PRETTY_PRINT = False
try:
    from terminaltables import SingleTable
    PRETTY_PRINT = True
except ImportError:
    print("terminaltabels module not installed. Printing will be ugly!\n")


sequences = []
stats = {}

if(len(argv) < 2):
    print("Please supply one or more folders as arguments.")
else:
    for arg in argv[1:]:
        sequences.extend([root for root, dirs, files in walk(arg) if not dirs])
    for seq in sequences:
        files = [n for n in listdir(seq) if isfile(join(seq,n))]
        tag = seq.split('/')[-2]
        minTimestamp = float('inf')
        maxTimestamp = 0
        for f in files:
            f = f.split('.')[0]
            timestamp = int(f)
            if timestamp > maxTimestamp:
                maxTimestamp = timestamp
            if timestamp < minTimestamp:
                minTimestamp = timestamp
        if tag not in stats.keys():
            stats[tag] = ([len(files)], [maxTimestamp-minTimestamp])
        else:
            (fileCount, timeSpan) = stats[tag]
            fileCount.append(len(files))
            timeSpan.append(maxTimestamp-minTimestamp)

    if stats:
        headers = []
        headers.append("Tag \nName")
        headers.append("Num. Seq.")
        headers.append("Frames/Sequence \nMean")
        headers.append("Frames/Sequence \nStd.")
        headers.append("Time/Sequence (ms) \nMean")
        headers.append("Time/Sequence (ms) \nStd.")
        tableData = [headers]
        totalSeq = 0
        for tag in stats.keys():
            content = []
            (fileCount, timeSpan) = stats[tag]
            meanFC = numpy.mean(fileCount)
            stdFC = numpy.std(fileCount)
            meanT = numpy.mean(timeSpan)
            stdT = numpy.std(timeSpan)
            totalSeq = totalSeq + len(fileCount)
            ## TODO - add outlier detection
            content.append([tag.upper(), str(len(fileCount)),
                            str(round(meanFC,2)), str(round(stdFC,2)),
                            str(int(meanT)), str(int(stdT))])
            tableData.extend(content)
        tableData.append(["T O T A L", str(totalSeq), '', '', '', ''])
        if PRETTY_PRINT:
            table = SingleTable(tableData, 'Raw Dataset Stats')
            # table.inner_row_border = True
            for i in range(1,6):
                table.justify_columns[i] = 'center'
            print(table.table)
        else:
            for header in tableData[0]:
                print(header.replace("\n","").ljust(25), end="")
            print("\n")
            for contents in tableData[1:]:
                for content in contents:
                    print(content.ljust(25), end="")
                print("\n")
