#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Jeniya Tabassum <jeniya.tabassum@gmail.com>

This evalutaion script is adapted from: https://github.com/sighsmile/conlleval 

# JT:208-09 added the evaluate_conll_file to take input in the f

This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.

IOB2:
- B = begin, 
- I = inside but not the first, 
- O = outside

e.g. 
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O

IOBES:
- B = begin, 
- E = end, 
- S = singleton, 
- I = inside but not the first or the last, 
- O = outside

e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O

prefix: IOBES
chunk_type: PER, LOC, etc.

#### Original Perl Script
# conlleval: evaluate result of processing CoNLL-2000 shared task
# usage:     conlleval [-l] [-r] [-d delimiterTag] [-o oTag] < file
#            README: http://cnts.uia.ac.be/conll2000/chunking/output.html
# options:   l: generate LaTeX output for tables like in
#               http://cnts.uia.ac.be/conll2003/ner/example.tex
#            r: accept raw result tags (without B- and I- prefix;
#                                       assumes one word per chunk)
#            d: alternative delimiter tag (default is white space or tab)
#            o: alternative outside tag (default is O)
# note:      the file should contain lines with items separated
#            by $delimiter characters (default space). The final
#            two items should contain the correct tag and the
#            guessed tag in that order. Sentences should be
#            separated from each other by empty lines or lines
#            with $boundary fields (default -X-).
# url:       http://lcg-www.uia.ac.be/conll2000/chunking/
# started:   1998-09-25
# version:   2004-01-26
# author of perl script:    Erik Tjong Kim Sang <erikt@uia.ua.ac.be>
# author of the main python script: sighsmile.github.io



"""
from __future__ import division, print_function, unicode_literals
import argparse
import sys
from collections import defaultdict


"""
• IOB1: I is a token inside a chunk, O is a token outside a chunk and B is the
beginning of chunk immediately following another chunk of the same Named Entity.
• IOB2: It is same as IOB1, except that a B tag is given for every token, which exists at
the beginning of the chunk.
• IOE1: An E tag used to mark the last token of a chunk immediately preceding another
chunk of the same named entity.
• IOE2: It is same as IOE1, except that an E tag is given for every token, which exists at
the end of the chunk.
• START/END: This consists of the tags B, E, I, S or O where S is used to represent a
chunk containing a single token. Chunks of length greater than or equal to two always
start with the B tag and end with the E tag.
• IO: Here, only the I and O labels are used. This therefore cannot distinguish between
adjacent chunks of the same named entity.

"""
# endOfChunk: checks if a chunk ended between the previous and current word
# arguments:  previous and current chunk tags, previous and current types
# note:       this code is capable of handling other chunk representations
#             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
def endOfChunk(prevTag, tag, prevType, type_):
    """
    checks if a chunk ended between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    """
    return ((prevTag == "B" and tag == "B") or
        (prevTag == "B" and tag == "O") or
        (prevTag == "I" and tag == "B") or
        (prevTag == "I" and tag == "O") or

        (prevTag == "E" and tag == "E") or
        (prevTag == "E" and tag == "I") or
        (prevTag == "E" and tag == "O") or
        (prevTag == "I" and tag == "O") or

        (prevTag != "O" and prevTag != "." and prevType != type_) or
        (prevTag == "]" or prevTag == "["))
        # corrected 1998-12-22: these chunks are assumed to have length 1


# startOfChunk: checks if a chunk started between the previous and current word
# arguments:    previous and current chunk tags, previous and current types
# note:         this code is capable of handling other chunk representations
#               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
def startOfChunk(prevTag, tag, prevType, type_):
    """
    checks if a chunk started between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    """
    chunkStart = ((prevTag == "B" and tag == "B") or
        (prevTag == "B" and tag == "B") or
        (prevTag == "I" and tag == "B") or
        (prevTag == "O" and tag == "B") or
        (prevTag == "O" and tag == "I") or

        (prevTag == "E" and tag == "E") or
        (prevTag == "E" and tag == "I") or
        (prevTag == "O" and tag == "E") or
        (prevTag == "O" and tag == "I") or

        (tag != "O" and tag != "." and prevType != type_) or
        (tag == "]" or tag == "["))
        # corrected 1998-12-22: these chunks are assumed to have length 1

    #print("startOfChunk?", prevTag, tag, prevType, type)
    #print(chunkStart)
    return chunkStart

def calcMetrics(TP, P, T, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = TP / P if P else 0
    recall = TP / T if T else 0
    FB1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * FB1
    else:
        return precision, recall, FB1

def splitTag(chunkTag, oTag = "O", raw = False):
    """
    Split chunk tag into IOB tag and chunk type;
    return (iob_tag, chunk_type)
    """
    if chunkTag == "O" or chunkTag == oTag:
        tag, type_ = "O", None
    elif raw:
        tag, type_ = "B", chunkTag
    else:
        try:
            # split on first hyphen, allowing hyphen in type
            tag, type_ = chunkTag.split('-', 1)
        except ValueError:
            tag, type_  = chunkTag, None
    return tag, type_

def countChunks(args,inputFile):
    """
    Process input in given format and count chunks using the last two columns;
    return correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter
    """
    boundary = "-X-"     # sentence boundary
    # delimiter = args.delimiter
    # raw = args.raw
    # oTag = args.oTag
    #inputFile=args.inputFile

    delimiter = args["delimiter"]
    raw = args["raw"]
    oTag = args["oTag"]

    fileIterator=open(inputFile)

    correctChunk = defaultdict(int)     # number of correctly identified chunks
    foundCorrect = defaultdict(int)     # number of chunks in corpus per type
    foundGuessed = defaultdict(int)     # number of identified chunks per type

    tokenCounter = 0     # token counter (ignores sentence breaks)
    correctTags = 0      # number of correct chunk tags

    lastType = None # temporary storage for detecting duplicates
    inCorrect = False # currently processed chunk is correct until now
    lastCorrect, lastCorrectType = "O", None    # previous chunk tag in corpus
    lastGuessed, lastGuessedType = "O", None  # previously identified chunk tag

    for line in fileIterator:
        # each non-empty line must contain >= 3 columns
        features = line.strip().split(delimiter)
        #print(features)
        if not features or features[0] == boundary:
            features = [boundary, "O", "O"]
        elif len(features) < 3:
             raise IOError("conlleval: unexpected number of features in line %s\n" % line)

        # extract tags from last 2 columns
        guessed, guessedType = splitTag(features[-1], oTag=oTag, raw=raw)
        correct, correctType = splitTag(features[-2], oTag=oTag, raw=raw)

        # 1999-06-26 sentence breaks should always be counted as out of chunk
        firstItem = features[0]
        if firstItem == boundary:
            guessed, guessedType = "O", None

        # decide whether current chunk is correct until now
        if inCorrect:
            endOfGuessed = endOfChunk(lastCorrect, correct, lastCorrectType, correctType)
            endOfCorrect = endOfChunk(lastGuessed, guessed, lastGuessedType, guessedType)
            if (endOfGuessed and endOfCorrect and lastGuessedType == lastCorrectType):
                inCorrect = False
                correctChunk[lastCorrectType] += 1
            elif ( endOfGuessed != endOfCorrect or guessedType != correctType):
                inCorrect = False

        startOfGuessed = startOfChunk(lastGuessed, guessed, lastGuessedType, guessedType)
        startOfCorrect = startOfChunk(lastCorrect, correct, lastCorrectType, correctType)
        if (startOfCorrect and startOfGuessed and guessedType == correctType):
            inCorrect = True
        if startOfCorrect:
            foundCorrect[correctType] += 1
        if startOfGuessed:
            foundGuessed[guessedType] += 1

        if firstItem != boundary:
            if correct == guessed and guessedType == correctType:
                correctTags += 1
            tokenCounter += 1

        lastGuessed, lastGuessedType = guessed, guessedType
        lastCorrect, lastCorrectType = correct, correctType

    if inCorrect:
        correctChunk[lastCorrectType] += 1

    return correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter

def evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter, latex=False, to_tsv=False, tsv_file_name="perf.tsv"):
    # sum counts
    # correctChunkSum = sum(correctChunk.values())
    # foundGuessedSum = sum(foundGuessed.values())
    # foundCorrectSum = sum(foundCorrect.values())

    correctChunkSum = sum(correctChunk[v] for v in correctChunk if v!="O")
    foundGuessedSum = sum(foundGuessed[v] for v in foundGuessed if v!="O")
    foundCorrectSum = sum(foundCorrect[v] for v in foundCorrect if v!="O")



    # sort chunk type names
    sortedTypes = list(foundCorrect) + list(foundGuessed)
    sortedTypes = list(set(sortedTypes))
    sortedTypes.sort()

    # print overall performance, and performance per chunk type
    eval_result={}
    
    # compute overall precision, recall and FB1 (default values are 0.0)
    precision, recall, FB1 = calcMetrics(correctChunkSum, foundGuessedSum, foundCorrectSum)
    result={}
    result["P"]=round(precision,2)
    result["R"]=round(recall,2)
    result["F1"]=round(FB1,2)
    result["Total Predicted"]= foundGuessedSum
    result["Correctly Predicted"]= correctChunkSum
    eval_result["overall"]=result
    eval_result["by_category"]={}


    
    # print overall performance
    print("processed %i tokens with %i phrases; " % (tokenCounter, foundCorrectSum), end='')
    print("found: %i phrases; correct: %i.\n" % (foundGuessedSum, correctChunkSum), end='')
    if tokenCounter:
        print("accuracy: %6.2f%%; " % (100*correctTags/tokenCounter), end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                (precision, recall, FB1))

    for i in sortedTypes:
        
        precision, recall, FB1 = calcMetrics(correctChunk[i], foundGuessed[i], foundCorrect[i])
        print("%17s: " %i , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                (precision, recall, FB1), end='')
        print(" foundGuessed:  %d" % foundGuessed[i])

        result={}
        
        result["P"]=round(precision,2)
        result["R"]=round(recall,2)
        result["F1"]=round(FB1,2)
        result["Total Predicted"]= foundGuessed[i]
        result["Correctly Predicted"]= correctChunk[i]

        eval_result["by_category"][i]=result

    # generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    if latex:
        print("        & Precision &  Recall  & F1 & Correct postive & Predicted positive & Gold positive \\\\\\hline", end='')
        for i in sortedTypes:
            precision, recall, FB1 = calcMetrics(correctChunk[i], foundGuessed[i], foundCorrect[i])
            print("\n%-7s &  %6.2f & %6.2f & %6.2f & %d & %d & %d \\\\" %
                 (i.replace("_"," "),precision,recall,FB1, correctChunk[i], foundGuessed[i],foundCorrect[i]), end='')
        print("\\hline")

        precision, recall, FB1 = calcMetrics(correctChunkSum, foundGuessedSum, foundCorrectSum)
        print("Overall &  %6.2f & %6.2f & %6.2f & %d & %d & %d  \\\\\\hline" %
              (precision,recall,FB1, correctChunkSum, foundGuessedSum, foundCorrectSum))
    if to_tsv:
        fout_tsv=open(tsv_file_name,"w")
        opline=""+"\t"+"Precision"+"\t"+"Recall"+"\t"+"F1"+"\n"
        fout_tsv.write(opline)
        for i in sortedTypes:
            precision, recall, FB1 = calcMetrics(correctChunk[i], foundGuessed[i], foundCorrect[i])
            opline= i.replace("_"," ") +"\t"+ str(precision) +"\t"+ str(recall) +"\t"+ str(FB1)+"\n"
            fout_tsv.write(opline)

        precision, recall, FB1 = calcMetrics(correctChunkSum, foundGuessedSum, foundCorrectSum)
        opline= "overall" +"\t"+ str(precision) +"\t"+ str(recall) +"\t"+ str(FB1)+"\n"
        fout_tsv.write(opline)
        fout_tsv.close()

    return eval_result


#JT : 2018-08
def evaluate_conll_file(inputFile="conll_output.txt", to_tsv=False, tsv_file_name="perf.tsv",latex=False, raw=False, delimiter=None, oTag="O"):
    #args = parse_args()
    args={}
    args["raw"]=raw
    args["latex"]=latex
    args["delimiter"]=delimiter
    args["oTag"]=oTag

    #print(type(args))
     # process input and count chunks
    correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter = countChunks(args,inputFile)

    # compute metrics and print
    eval_result = evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter, latex=latex, to_tsv=to_tsv, tsv_file_name=tsv_file_name)
    return eval_result

if __name__ == "__main__":
    eval_result = evaluate_conll_file(inputFile="so_output_w_window_all.txt")
    print(eval_result)
    #sys.exit(0)
