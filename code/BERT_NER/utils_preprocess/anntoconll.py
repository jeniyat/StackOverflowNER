#!/usr/bin/env python

# Convert text and standoff annotations into CoNLL format.

from __future__ import print_function

import os
from glob import glob
import re
import sys
from collections import namedtuple
from io import StringIO
from os import path

import sys
from os.path import join as path_join
from os.path import dirname
from sys import path as sys_path



# assume script in brat tools/ directory, extend path to find sentencesplit.py
sys_path.append(path_join(dirname(__file__), '.'))

sys.path.append('.')


from sentencesplit import sentencebreaks_to_newlines


options = None

EMPTY_LINE_RE = re.compile(r'^\s*$')
CONLL_LINE_RE = re.compile(r'^\S+\t\d+\t\d+.')

 
import stokenizer #JT: Dec 6

import ftfy #JT: Feb 20


from map_text_to_char import map_text_to_char  #JT: Dec 6


def argparser():
    import argparse

    ap = argparse.ArgumentParser(description='Convert text and standoff ' +
                                 'annotations into CoNLL format.')
    ap.add_argument('-a', '--annsuffix', default=".ann",
                    help='Standoff annotation file suffix (default "ann")')
    ap.add_argument('-c', '--singleclass', default=None,
                    help='Use given single class for annotations')
    ap.add_argument('-n', '--nosplit', default=True, action='store_true',
                    help='No sentence splitting')
    ap.add_argument('-o', '--outsuffix', default="conll",
                    help='Suffix to add to output files (default "conll")')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Verbose output')
    # ap.add_argument('text', metavar='TEXT', nargs='+',
    #                 help='Text files ("-" for STDIN)')
    return ap


def read_sentence(f):
    """Return lines for one sentence from the CoNLL-formatted file.

    Sentences are delimited by empty lines.
    """

    lines = []
    for l in f:
        lines.append(l)
        if EMPTY_LINE_RE.match(l):
            break
        if not CONLL_LINE_RE.search(l):
            raise FormatError(
                'Line not in CoNLL format: "%s"' %
                l.rstrip('\n'))
    return lines



def strip_labels(lines):
    """Given CoNLL-format lines, strip the label (first TAB-separated field)
    from each non-empty line.

    Return list of labels and list of lines without labels. Returned
    list of labels contains None for each empty line in the input.
    """

    labels, stripped = [], []

    labels = []
    for l in lines:
        if EMPTY_LINE_RE.match(l):
            labels.append(None)
            stripped.append(l)
        else:
            fields = l.split('\t')
            labels.append(fields[0])
            stripped.append('\t'.join(fields[1:]))

    return labels, stripped



def attach_labels(labels, lines):
    """Given a list of labels and CoNLL-format lines, affix TAB-separated label
    to each non-empty line.

    Returns list of lines with attached labels.
    """

    assert len(labels) == len(
        lines), "Number of labels (%d) does not match number of lines (%d)" % (len(labels), len(lines))

    attached = []
    for label, line in zip(labels, lines):
        empty = EMPTY_LINE_RE.match(line)
        assert (label is None and empty) or (label is not None and not empty)

        if empty:
            attached.append(line)
        else:
            attached.append('%s\t%s' % (label, line))

    return attached



def text_to_conll(f):
    """Convert plain text into CoNLL format."""
    global options

    if options.nosplit:
        sentences = f.readlines()
    else:
        sentences = []
        for l in f:
            l = sentencebreaks_to_newlines(l)
            sentences.extend([s for s in NEWLINE_TERM_REGEX.split(l) if s])

    lines = []

    offset = 0
    # print(sentences)
    #JT: Feb 19: added it for resolving char encoding issues
    fixed_sentences = []
    for s in sentences:
        # print(s)
        # fixed_s = ftfy.fix_text(s)
        # # print(fixed_s)
        # fixed_sentences.append(fixed_s)
        fixed_sentences.append(s)

    # for s in sentences:
    for s in fixed_sentences:
        nonspace_token_seen = False
        # print(s)
        
        
        
        try:
            tokens = stokenizer.tokenize(s)
        except stokenizer.TimedOutExc as e:
            try:
                print("***********using ark tokenizer")
                tokens = ark_twokenize.tokenizeRawTweetText(s)
            except Exception as e:
                    print(e)
        # print("tokens: ", tokens)
        token_w_pos = map_text_to_char(s, tokens, offset)
        # print("token_w_pos: ",token_w_pos)

        for(t, pos) in token_w_pos:
            if not t.isspace():
                lines.append(['O', pos, pos + len(t), t])
        
        lines.append([])

        offset+=len(s)


        # tokens = [t for t in TOKENIZATION_REGEX.split(s) if t] # JT : Dec 6
        # for t in tokens:
        #     if not t.isspace():
        #         lines.append(['O', offset, offset + len(t), t])
        #         nonspace_token_seen = True
        #     offset += len(t)

        # # sentences delimited by empty lines
        # if nonspace_token_seen:
        #     lines.append([])

    # add labels (other than 'O') from standoff annotation if specified
    if options.annsuffix:
        lines = relabel(lines, get_annotations(f.name), f)

    # lines = [[l[0], str(l[1]), str(l[2]), l[3]] if l else l for l in lines] #JT: Dec 6
    lines = [[l[3],l[0]] if l else l for l in lines] #JT: Dec 6
    return StringIO('\n'.join(('\t'.join(l) for l in lines)))


def relabel(lines, annotations, file_name):
    # print("lines: ",lines)
    # print("annotations", annotations)
    global options

    # TODO: this could be done more neatly/efficiently
    offset_label = {}

    for tb in annotations:
        for i in range(tb.start, tb.end):
            if i in offset_label:
                print("Warning: overlapping annotations in ", file=sys.stderr)
            offset_label[i] = tb

    prev_label = None
    for i, l in enumerate(lines):
        if not l:
            prev_label = None
            continue
        tag, start, end, token = l

        # TODO: warn for multiple, detailed info for non-initial
        label = None
        for o in range(start, end):
            if o in offset_label:
                if o != start:
                    print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                        token, offset_label[o].text), file=sys.stderr)
                label = offset_label[o].type
                break

        if label is not None:
            if label == prev_label:
                tag = 'I-' + label
            else:
                tag = 'B-' + label
        prev_label = label

        lines[i] = [tag, start, end, token]

    # optional single-classing
    if options.singleclass:
        for l in lines:
            if l and l[0] != 'O':
                l[0] = l[0][:2] + options.singleclass

    return lines




def process_files(files, output_directory, phase_name=""):
    global options
    # print("phase_name: ",phase_name)

    nersuite_proc = []

    
    for fn in sorted(files):
        # print("now_processing: ",fn)
        with open(fn, 'rU') as f:
            try:
                lines = text_to_conll(f)
            except:
                continue

            # TODO: better error handling
            if lines is None:
                print("Line is None")
                continue
            file_name=fn.split("/")[-1][0:-4]
            ofn = output_directory+file_name+"_" +options.outsuffix.replace(".","")+"_"+phase_name.replace("/","")+".txt"
            with open(ofn, 'wt') as of:
                of.write(''.join(lines))

         
TEXTBOUND_LINE_RE = re.compile(r'^T\d+\t')

Textbound = namedtuple('Textbound', 'start end type text')


def parse_textbounds(f):
    """Parse textbound annotations in input, returning a list of Textbound."""

    textbounds = []

    for l in f:
        l = l.rstrip('\n')

        if not TEXTBOUND_LINE_RE.search(l):
            continue

        id_, type_offsets, text = l.split('\t')
        type_, start, end = type_offsets.split()
        start, end = int(start), int(end)

        textbounds.append(Textbound(start, end, type_, text))

    return textbounds


def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check
    for t1 in textbounds:
        for t2 in textbounds:
            if t1 is t2:
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # eliminate shorter
            if t1.end - t1.start > t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[t2] = True
            else:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[t1] = True

    return [t for t in textbounds if t not in eliminate]


def get_annotations(fn):
    global options

    annfn = path.splitext(fn)[0] + options.annsuffix

    with open(annfn, 'rU') as f:
        textbounds = parse_textbounds(f)

    textbounds = eliminate_overlaps(textbounds)

    return textbounds


def Read_Main_Input_Folder(input_folder):
    start_dir = input_folder
    
    pattern   = "*.txt"
    file_location_list=[]
    for dir,_,_ in os.walk(start_dir):
        file_location_list.extend(glob(os.path.join(dir,pattern))) 
    
    return file_location_list



def process_folder(source_folder, output_dir_ann, min_folder_number = 1, max_folder_number=10 ):
    # for i in range(min_folder_number,max_folder_number+1):
        # for j in range(1,6):
            # phase_name="phase_"+str(i).zfill(2) + "_"+str(j).zfill(2)+"/"
    input_folder=source_folder
    print(input_folder)
    list_of_files=Read_Main_Input_Folder(input_folder)
    process_files(list_of_files, output_dir_ann)


def convert_standoff_to_conll(source_directory_ann, output_directory_conll):
    global options
    

    
    argv = sys.argv

    
    options = argparser().parse_args(argv[1:])


    # print(options)
    # sorce_folder = "checked_annotation/"
    # phase_name="phase_02_05/"
    # input_folder=sorce_folder+phase_name
    # list_of_files=Read_Main_Input_Folder(input_folder)
    # output_dir_ann = "Conlll_Output_ANN/"
    # process_files(list_of_files, output_dir_ann, phase_name)

    

    
    process_folder(source_directory_ann, output_directory_conll)
    

    


    # sorce_folder = "raw_data/"
    # output_dir_raw = "Conlll_Output_RAW/"


    # sorce_folder = "raw_data/"
    # phase_name="phase_02_05/"
    # input_folder=sorce_folder+phase_name
    # list_of_files=Read_Main_Input_Folder(input_folder)
    # output_dir_ann = "Conlll_Output_RAW/"
    # process_files(list_of_files, output_dir_ann, phase_name)








if __name__ == '__main__':
    source_directory_ann = "../temp_files/standoff_files/"
    output_directory_conll = "../temp_files/conll_files/"
    convert_standoff_to_conll(source_directory_ann, output_directory_conll)
	















