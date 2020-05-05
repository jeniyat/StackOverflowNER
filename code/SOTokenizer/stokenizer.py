#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Jeniya Tabassum <jeniya.tabassum@gmail.com>

stokenizer -- a tokenizer designed for StackOverflow text.

This tokenizer is build on top of a tweet tokenizer : https://github.com/myleott/ark-twokenize-py/blob/master/twokenize.py
We updated the rules in tweet tokenizer to adjust with the software domain texts.

Below is the history of the development fo the  tweet tokenizer
(1) Brendan O'Connor wrote original version in Python, http://github.com/brendano/tweetmotif
       TweetMotif: Exploratory Search and Topic Summarization for Twitter.
       Brendan O'Connor, Michel Krieger, and David Ahn.
       ICWSM-2010 (demo track), http://brenocon.com/oconnor_krieger_ahn.icwsm2010.tweetmotif.pdf
(2a) Kevin Gimpel and Daniel Mills modified it for POS tagging for the CMU ARK Twitter POS Tagger
(2b) Jason Baldridge and David Snyder ported it to Scala
(3) Brendan bugfixed the Scala port and merged with POS-specific changes
    for the CMU ARK Twitter POS Tagger
(4) Tobi Owoputi ported it back to Java and added many improvements (2012-06)
(5) Myle Ott ported it to Python3 .


"""
from __future__ import unicode_literals

import operator
import re
import sys

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

try:
    import html
except ImportError:
    pass

#items="a|b => regex_or(items) ==> (?:a|b)
def regex_or(*items):
    return '(?:' + '|'.join(items) + ')'


Contractions = re.compile(u"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$", re.UNICODE)
Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

punctChars = r"['\"“”‘’.?!…,:;]"
#punctSeq   = punctChars+"+"  #'anthem'. => ' anthem '.
punctSeq   = r"['\"“”‘’]+|[.?!,…]+|[:;]+" #'anthem'. => ' anthem ' .
entity     = r"&(?:amp|lt|gt|quot);" #html tag entities &quot; => "
#  URLs


# BTO 2012-06: everyone thinks the daringfireball regex should be better, but they're wrong.
# If you actually empirically test it the results are bad.
# Please see https://github.com/brendano/ark-tweet-nlp/pull/9

urlStart1  = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx|aspx)"
ccTLDs   = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)" #TODO: remove obscure country domains?
urlStart2  = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, ccTLDs) + r"(?:\."+ccTLDs+r")?(?=\W|$)"
urlBody    = r"(?:[^\.\s<>][^\s<>]*?)?"
urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
urlEnd     = r"(?:\.\.+|[<>]|\s|$)"
url        = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")"


# Numeric
timeLike   = r"\d+(?::\d+){1,2}"
#numNum     = r"\d+\.\d+"
numberWithCommas = r"(?:(?<!\d)\d{1,3},)+?\d{3}" + r"(?=(?:[^,\d]|$))"
numComb  = u"[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6\u00a2-\u00a5\u20a0-\u20b9]?\\d+(?:\\.\\d+)+%?"

# Abbreviations
boundaryNotDot = regex_or("$", r"\s", r"[“\"?!,:;]", entity)
aa1  = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
aa2  = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
arbitraryAbbrev = regex_or(aa1, aa2, standardAbbreviations)
separators  = "(?:--+|―|—|~|–|=)"
decorations = u"(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)"
thingsThatSplitWords = r"[^\s\.,?\"]"
embeddedApostrophe = thingsThatSplitWords+r"+['’′]" + thingsThatSplitWords + "*"

#  Emoticons
# myleott: in Python the (?iu) flags affect the whole expression
#normalEyes = "(?iu)[:=]" # 8 and x are eyes but cause problems
normalEyes = "[:=]" # 8 and x are eyes but cause problems
wink = "[;]"
noseArea = "(?:|-|[^a-zA-Z0-9 ])" # doesn't get :'-(
happyMouths = r"[D\)\]\}]+"
sadMouths = r"[\(\[\{]+"
tongue = "[pPd3]+"
otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)" # remove forward slash if http://'s aren't cleaned

# mouth repetition examples:
# @aliciakeys Put it in a love song :-))
# @hellocalyclops =))=))=)) Oh well

# myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails
#bfLeft = u"(♥|0|o|°|v|\\$|t|x|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
bfLeft = u"(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)"
bfCenter = r"(?:[\.]|[_-]+)"
bfRight = r"\2"
s3 = r"(?:--['\"])"
s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
s5 = "(?:[.][_]+[.])"
# myleott: in Python the (?i) flag affects the whole expression
#basicface = "(?:(?i)" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5
basicface = "(?:" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5

eeLeft = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
eeRight= u"[\\-=\\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+"
eeSymbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight

oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"


emoticon = regex_or(
        # Standard version  :) :( :] :D :P
        "(?:>|&gt;)?" + regex_or(normalEyes, wink) + regex_or(noseArea,"[Oo]") + regex_or(tongue+r"(?=\W|$|RT|rt|Rt)", otherMouths+r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),

        # reversed version (: D:  use positive lookbehind to remove "(word):"
        # because eyes on the right side is more ambiguous with the standard usage of : ;
        regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sadMouths,happyMouths,otherMouths) + noseArea + regex_or(normalEyes, wink) + "(?:<|&lt;)?",

        #inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style
        eastEmote.replace("2", "1", 1), basicface,
        # iOS 'emoji' characters (some smileys, some symbols) [\ue001-\uebbb]
        # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this

        # myleott: o.O and O.o are two of the biggest sources of differences
        #          between this and the Java version. One little hack won't hurt...
        oOEmote
)

Hearts = "(?:<+/?3+)+" #the other hearts are in decorations

Arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", u"[\u2190-\u21ff]+")

# BTO 2011-06: restored Hashtag, AtMention protection (dropped in original scala port) because it fixes
# "hello (#hashtag)" ==> "hello (#hashtag )"  WRONG
# "hello (#hashtag)" ==> "hello ( #hashtag )"  RIGHT
# "hello (@person)" ==> "hello (@person )"  WRONG
# "hello (@person)" ==> "hello ( @person )"  RIGHT
# ... Some sort of weird interaction with edgepunct I guess, because edgepunct
# has poor content-symbol detection.

# This also gets #1 #40 which probably aren't hashtags .. but good as tokens.
# If you want good hashtag identification, use a different regex.
Hashtag = "#[a-zA-Z0-9_]+"  #optional: lookbehind for \b
#optional: lookbehind for \b, max length 15
AtMention = "[@＠][a-zA-Z0-9_]+"

# I was worried this would conflict with at-mentions
# but seems ok in sample of 5800: 7 changes all email fixes
# http://www.regular-expressions.info/email.html
Bound = r"(?:\W|^|$)"
Email = regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" +Bound+")"

# JT's additions to rules are as the following: 
# File_Ext, Path, OR_Words, File_Path_W_File_Name, Windows Path, Class_Name, Func_Name, Func_Name_Recursive, Class_Func_Name, HTML_Tag, Comparison Operation, Numbered_List, SPECIAL_WORDS

# JT: 2018-08
# '.html.erb' => '. html . erb' :: WRONG
# '.html.erb' => '.html.erb' :: WRONG
File_Ext = r"[.]?[\w.\-]*\.[\w]+(?=" +Bound+")"

# JT: 2018-08
Path=r'(?:/?[\w\-.]+\/+)+'
#Path=r'(?:/?[\w\-.]+\/+[\w\-.]*)+'

# JT: 2018-08
# '/templates/your_template/.php.ini' => '/templates/your_template/ . php . ini' :: WRONG
# '/templates/your_template/.php.ini' => '/templates/your_template/.php.ini' :: RIGHT
File_Path_W_File_Name=Path +"(?:" + File_Ext + ")*"


# JT: 2018-08
# 'GNU/LINUX' => 'GNU / LINUX' :: WRONG
# 'GNU/LINUX' => 'GNU/LINUX' :: RIGHT
OR_Words=r'([\w\-.:]*\/[\w\.:]*)+'+"(?=" +Bound+")"


# JT: 2018-08
# 'd:\\ProgramFiles\x07dtbundle' => 'd : \\ProgramFiles\x07dtbundle' :: WRONG
# 'd:\\ProgramFiles\x07dtbundle' => 'd:\\ProgramFiles\x07dtbundle' :: RIGHT
Windows_Path=r'((?:(?:[a-zA-Z]:)?\\)[\\\S|*\S]?\S*)'+"(?=" +Bound+")"


# JT: 2018-08
# 'javax.swing.Timer' => 'javax . swing . Timer' :: WRONG
# 'javax.swing.Timer' => 'javax.swing.Timer' :: RIGHT
Class_Name  =r"[\w.:\-\>]*[\.:\-\>][\w\*]*(?=" +Bound+")"

# JT: 2018-08
#Func_Name =  r"([\w@\-]+\((?:[\w@\-]+(?:,\s*)?)*\))"+"(?=" +Bound+")" #-----this will not allow space and quote in function argument
Func_Name = r"([\w@\-]+\((?:[\$\w@\-\'\"\s?]+(?:,\s*)?)*\))"+"(?=" +Bound+")"

# JT: 2018-08
Func_Name_Recursive = r"([\w@\-.\(\)]+\((?:[\$\w@\-\'\"\s?]+(?:,\s*)?)*\))"+"(?=" +Bound+")"
#Func_Name_Recursive = r"([$\w@\-.\(\)]+\((?:[\$\w@\-\'\"\s?]+(?:,\s*)?)*\)(?:[.][\w@\-]*)*)"+"(?=" +Bound+")" #last part is included for words like $http.get().success
#Func_Name_Recursive = r"([\w@\-.\(\)]+\((?:[\$\w@\-\'\"\s?]+(?:,\s*)?)*\)(?:[.][\w@\-]*)*)"+"(?=" +Bound+")"

# JT: 2018-08
# 'txScope.Complete()' => 'txScope.Complete(arg1, arg2)' :: RIGHT
# 'txScope.Complete()' => 'txScope . Complete(arg1 , arg2 )' :: WRONG
#Class_Func_Name = r"([\w.:\-\>]*[\.:\-\>][\w]*\((?:[\w@\-]+[\.:\-\>\s=]*[\w]*(?:,\s*)?)*\))"+"(?=" +Bound+")" #-----this will not allow space and quote in function argument
Class_Func_Name = r"([\$\w.:\-\>]*[\.:\-\>][\w\$]*\((?:[\$\w@\-\"\'\s]+[\.:\-\>\s=]*[\$\w]*(?:,\s*)?)*\))"+"(?=" +Bound+")" 

# JT: 2018-08
# '<%=@consumer.name%>' => '<%= @consumer . name% >' :: WRONG
# '<%=@consumer.name%>' => '<%=@consumer.name%>' :: RIGHT
HTML_Tag=r"<.*>"+"(?=" +Bound+")"

# JT: 2018-08
# '==' => "= =" :: WRONG
# '==' => "==" :: RIGHT
Comparison_Operators=r"==|!=|<=|>=|:="

# JT: 2018-08
# num2roman(num) and  generate_number_list(limit) are the function for genrating the numbered list in roman and english numerics, followed by )
def num2roman(num):
    num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

    roman = ''

    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i

    return roman

# JT: 2018-08
def generate_number_list(limit):
    numbered_lists_joined_string=r""
    for i in range(1,limit+1):
        roman_upper_case = num2roman(i)
        roman_lower_case = roman_upper_case.lower()
        numeric_str= str(i)
        current_string=roman_upper_case+"\)|"+roman_lower_case+"\)|"+numeric_str+"\)|"
        if i==limit:
            current_string=current_string[:-1]
        
        numbered_lists_joined_string+=current_string
    return numbered_lists_joined_string




# Numbered_List for genrating the numbered list in roman and english numerics, followed by )  and we protect these number from splitting
# "ii)" => "ii ) " :: WRONG
# "ii)" => "ii) " :: RIGHT

#set upto which number we want to generate the numebered list, for now we set this value to 5, we can increase it later
Upper_Limit_For_Numbered_List=5
Numbered_List=generate_number_list(Upper_Limit_For_Numbered_List)+"(?=" +Bound+")" 


# JT: 2018-08
# Additionally, these words are "protected", meaning they shouldn't be further split themselves.
SPECIAL_WORDS=r"^http:|^HTTP:|^vs.|^c#.net|^C#.net"+"(?=" +Bound+")"

# We will be tokenizing using these regexps as delimiters
# Additionally, these things are "protected", meaning they shouldn't be further split themselves.
Protected  = re.compile(
    regex_or(
        Func_Name_Recursive, #JT
        Hearts,
        url,
        Email,
        SPECIAL_WORDS, #JT
        OR_Words, #JT
        Numbered_List, #JT
        Windows_Path, #JT
        Class_Func_Name, #JT
        emoticon,
        Func_Name, #JT
        Comparison_Operators, #JT
        Class_Name, #JT
        File_Path_W_File_Name, #JT
        #OR_Words, #JT
        File_Ext, #JT
        Hashtag,
        #HTML_Tag, #JT
        Path, #JT
        timeLike,
        #numNum,
        numberWithCommas,
        numComb,
        emoticon,
        Arrows,
        entity,
        punctSeq,
        arbitraryAbbrev,
        separators,
        decorations,
        embeddedApostrophe,
        AtMention), re.UNICODE)


# Edge punctuation
# Want: 'foo' => ' foo '
# While also:   don't => don't
# the first is considered "edge punctuation".
# the second is word-internal punctuation -- don't want to mess with it.
# BTO (2011-06): the edgepunct system seems to be the #1 source of problems these days.
# I remember it causing lots of trouble in the past as well.  Would be good to revisit or eliminate.

# Note the 'smart quotes' (http://en.wikipedia.org/wiki/Smart_quotes)
#edgePunctChars    = r"'\"“”‘’«»{}\(\)\[\]\*&" #add \\p{So}? (symbols)
edgePunctChars    = u"'\"“”‘’«»{}\\(\\)\\[\\]\\*&" #add \\p{So}? (symbols)
edgePunct    = "[" + edgePunctChars + "]"
notEdgePunct = "[a-zA-Z0-9]" # content characters
offEdge = r"(^|$|:|;|\s|\.|,)"  # colon here gets "(hello):" ==> "( hello ):"
EdgePunctLeft  = re.compile(offEdge + "("+edgePunct+"+)("+notEdgePunct+")", re.UNICODE)
EdgePunctRight = re.compile("("+notEdgePunct+")("+edgePunct+"+)" + offEdge, re.UNICODE)

#JT: splitEdgePunct_software is adapted from splitEdgePunct function of tweet tokenizer 
#JT: we  donot have to add Func Rule here as Func Rule are not spiltted by edgePucnLeft & edgePucnRight
def splitEdgePunct_software(input):
    ##print("splitEdgePunct_software input: ",input)
    #first find the word identified by Func_Name_Recursive
    Func_Name_Recursive_Rule=re.compile(Func_Name_Recursive)
    Func_Name_Recursive_Words = Func_Name_Recursive_Rule.findall(input)

    ##print("Func_Name_Recursive_Words: ", Func_Name_Recursive_Words)


    #remove white space from the words identified by Func_Name_Recursive_Rule & save them to Func_Name_Recursive_Words_modified
    Func_Name_Recursive_Words_modified=[]
    Func_Name_Recursive_Words_white_space_removed=[]
    

    for w in Func_Name_Recursive_Words:
        w_ = re.sub(r"[\s]", '', w)
        Func_Name_Recursive_Words_white_space_removed.append((w,w_))
        Func_Name_Recursive_Words_modified.append(w_)
    
    #remove white space from the main input for those words which are identified by Func_Name_Recursive_Func_Rule
    for w in Func_Name_Recursive_Words_white_space_removed:
        main_string = w[0]
        space_removed_string = w[1]
        input=input.replace(main_string,space_removed_string)

    ##print("input: after removing spaces Func_Name_Recursive_Words_white_space_removed: ",input)
    #first find the word identified by Class_Func_Rule
    Class_Func_Rule=re.compile(Class_Func_Name)
    Class_Func_Words = Class_Func_Rule.findall(input)


    #remove white space from the words identified by Class_Func_Rule & save them to Class_Func_Words_modified
    Class_Func_Words_modified=[]
    Class_Func_Words_white_space_removed=[]
    

    for w in Class_Func_Words:
        w_ = re.sub(r"[\s]", '', w)
        Class_Func_Words_white_space_removed.append((w,w_))
        Class_Func_Words_modified.append(w_)
    
    #remove white space from the main input for those words which are identified by Class_Func_Rule
    for w in Class_Func_Words_white_space_removed:
        main_string = w[0]
        space_removed_string = w[1]
        input=input.replace(main_string,space_removed_string)


    #now split sentence by white space
    split_input=input.split()
    

    
    ##print("Func_Name_Recursive_Words_modified: ",Func_Name_Recursive_Words_modified )
    ##print("split input: ",split_input)

   
    
    #apply EdgePunct rules on each word
    special_words=["vs.","http:"]
    output_tokens=[]
    for word_main in split_input:
        if word_main.lower() in special_words:
            split_words=[word_main]
        else:
            split_words=Split_End_of_Sentence_Punc([word_main])

        for word in split_words:
            if word in Func_Name_Recursive_Words_modified:
                ##print("word found: ",word)
                output_tokens.append(word)
            elif word in Class_Func_Words_modified:
                output_tokens.append(word)
            else:
              word = EdgePunctLeft.sub(r"\1\2 \3", word)  
              word = EdgePunctRight.sub(r"\1\2 \3", word) 
              output_tokens.append(word)


    #join the extracted words
    # 
    modified_input=" ".join(output_tokens)
    ##print("modified_input: ",modified_input)
    ##print("\n\n")

    #add the white space in the function back to revert to original input
    for w in Func_Name_Recursive_Words_white_space_removed:
        main_string = w[0]
        space_removed_string = w[1]
        modified_input=modified_input.replace(space_removed_string, main_string)


    #add the white space in the function back to revert to original input
    for w in Class_Func_Words_white_space_removed:
        main_string = w[0]
        space_removed_string = w[1]
        modified_input=modified_input.replace(space_removed_string, main_string)


    

   

    ##print("modified_input: ",modified_input)

    return modified_input


# JT: 2018-08: Split_End_of_Sentence_Punc(token_list) split the punctuation letter form the last token
# 'Update the sdk version.' => 'Update the sdk version . '

def Split_End_of_Sentence_Punc(token_list):
    if len(token_list[-1])==1:
        return token_list

    arbitraryAbbrev_rule=re.compile(arbitraryAbbrev)
    list_of_abbrev= arbitraryAbbrev_rule.findall(token_list[-1])
    if len(list_of_abbrev)>0:
        return token_list

    list_of_punctuations=[".",":","?",";","-","!"]
    
    
    end_token=[s for s in list_of_punctuations if s==token_list[-1][-1] and s!=token_list[-1][-2]]
    ##print("end_token: ",end_token)
    #if token_list[-1][-1]==".":
    if len(end_token)==1:
        end_punc_char=end_token[0]
        token_list[-1]=token_list[-1][:-1]
        token_list.append(end_punc_char)
    return token_list







# simpleTokenize_software is adapted from simpleTokenize function of the tweet tokenizaer only 

def simpleTokenize_software(text):

    # JT:2018-08:  Do the software domain specific splitting 
    splitPunctText = splitEdgePunct_software(text)

    textLength = len(splitPunctText)
    ##print("splitPunctText: ",splitPunctText)

    # BTO: the logic here got quite convoluted via the Scala porting detour
    # It would be good to switch back to a nice simple procedural style like in the Python version
    # ... Scala is such a pain.  Never again.

    # Find the matches for subsequences that should be protected,
    # e.g. URLs, 1.0, U.N.K.L.E., 12:53
    bads = []
    badSpans = []

    for match in Protected.finditer(splitPunctText):
        # The spans of the "bads" should not be split.
        ##print(match)
        if (match.start() != match.end()): #unnecessary?
            bads.append( [splitPunctText[match.start():match.end()]] )
            badSpans.append( (match.start(), match.end()) )
    ##print("bads: ",bads)
    # Create a list of indices to create the "goods", which can be
    # split. We are taking "bad" spans like
    #     List((2,5), (8,10))
    # to create
    #     List(0, 2, 5, 8, 10, 12)
    # where, e.g., "12" here would be the textLength
    # has an even length and no indices are the same
    indices = [0]
    for (first, second) in badSpans:
        indices.append(first)
        indices.append(second)
    indices.append(textLength)

    # Group the indices and map them to their respective portion of the string
    splitGoods = []
    for i in range(0, len(indices), 2):
        goodstr = splitPunctText[indices[i]:indices[i+1]]
        splitstr = goodstr.strip().split(" ")
        splitGoods.append(splitstr)

    #  Reinterpolate the 'good' and 'bad' Lists, ensuring that
    #  additonal tokens from last good item get included
    zippedStr = []
    for i in range(len(bads)):
        zippedStr = addAllnonempty(zippedStr, splitGoods[i])
        zippedStr = addAllnonempty(zippedStr, bads[i])
    zippedStr = addAllnonempty(zippedStr, splitGoods[len(bads)])

    # BTO: our POS tagger wants "ur" and "you're" to both be one token.
    # Uncomment to get "you 're"
    splitStr = []
    for tok in zippedStr:
       splitStr.extend(splitToken(tok))
    zippedStr = splitStr
    ##print("zippedStr: ",zippedStr)
    return zippedStr

    #zippedStr_updated=Split_End_of_Sentence_Punc(zippedStr)

    # #print("zippedStr", zippedStr)
    # #print("zippedStr_updated", zippedStr_updated)

    #return zippedStr_updated



def addAllnonempty(master, smaller):
    for s in smaller:
        strim = s.strip()
        if (len(strim) > 0):
            master.append(strim)
    return master

# "foo   bar " => "foo bar"
def squeezeWhitespace(input):
    return Whitespace.sub(" ", input).strip()

# Final pass tokenization based on special patterns
def splitToken(token):
    m = Contractions.search(token)
    if m:
        return [m.group(1), m.group(2)]
    return [token]

# Assume 'text' has no HTML escaping.
def tokenize_text(text):
    return simpleTokenize_software(squeezeWhitespace(text))


# Twitter text comes HTML-escaped, so unescape it.
# We also first unescape &amp;'s, in case the text has been buggily double-escaped.
def normalizeTextForTagger(text):
    assert sys.version_info[0] >= 3 and sys.version_info[1] > 3, 'Python version >3.3 required'
    text = text.replace("&amp;", "&")
    text = html.unescape(text)
    return text


# JT: 2018-08:  Split_On_Multiple_Dot(input_word), 
# 'the queries....it' => "queries .... it"
def Split_On_Multiple_Dot(input_word):
    multiple_dot=r'\w*[.][.]+\w*'
    multiple_dot_rule=re.compile(multiple_dot)
    words_with_multiple_dots = multiple_dot_rule.findall(input_word)

    new_tokens=[]
    for word in words_with_multiple_dots:
        splitter=""
        for i in range(0,word.count(".")):
            splitter+="."
        word_splitted = word.split(splitter)
        
        split_word_index=0
        for split_word in word_splitted:
            if split_word =="":continue
            if split_word_index>0:
                new_tokens.append(splitter)
            new_tokens.append(split_word)
            split_word_index+=1
        if split_word_index==1:
            new_tokens.append(splitter)
    return new_tokens

# JT:2018-08: Split_On_Non_function_end_parenthesis(input_word) splits the surrounding parenthesis of a word only if the word is not a function
# '{"kind"=>"GGG"}' =>>  ' { "kind"=>"GGG" } '
def Split_On_Non_function_end_parenthesis(input_word):
    ##print("input_word for Split_On_Non_function_end_parenthesis:", input_word)
    new_token=[input_word]

    if len(input_word)==1:
        return new_token

    #if the word is from a numbered list do not split end bracket
    numbered_list_rule=re.compile(Numbered_List)
    numbered_list_words = numbered_list_rule.findall(input_word)

    if len(numbered_list_words)>0:
        return new_token

    #if the word is from a emoticon do not split end bracket
    emoticon_rule=re.compile(emoticon)
    emoticon_words = emoticon_rule.findall(input_word)

    if len(emoticon_words)>0:
        return new_token

    #if the word is from a class function do not split end bracket
    class_func_name_rule=re.compile(Class_Func_Name)
    class_func_words = class_func_name_rule.findall(input_word)

    if len(class_func_words)>0:
        return new_token

    #if the word is from a function do not split end bracket
    func_name_rule=re.compile(Func_Name)
    func_words = func_name_rule.findall(input_word)
    if len(func_words)>0:
        return new_token

    #otherwise check if there is any end bracket, if then add space infront of it
    if ")" in input_word and "(" not in input_word:
        input_word_updated=input_word.replace(")"," )")
        new_token=[input_word_updated]
    elif "(" in input_word and ")" not in input_word:
        input_word_updated=input_word.replace("(","( ")
        new_token=[input_word_updated]

    elif "]" in input_word and "[" not in input_word:
        input_word_updated=input_word.replace("]"," ]")
        new_token=[input_word_updated]

    elif "[" in input_word and "]" not in input_word:
        input_word_updated=input_word.replace("[","[ ")
        new_token=[input_word_updated]
    
    return new_token

# JT: 2018-08: Split_On_last_letter_Colon_Mark(input_word) will create a spcace between Quote and letter
# ' Example:"'=>'Example :"'
def Split_On_last_letter_Colon_Mark(input_word):
    ##print("inside Split_On_last_letter_Punc_Mark", input_word)
    new_token=[input_word]

    sp_rule=re.compile(SPECIAL_WORDS)
    sp_rule_words=sp_rule.findall(input_word)
    if len(sp_rule_words)>0:
        return new_token

    
    if len(input_word)==1:
        return new_token

    #if count of ":" is more than word then it is probably a word mentioning a class name, so then we are not going to split it
    if input_word.count(":")>1:
        return new_token
    #if is not a class name, then we are going to add space before token, Netbeans: >> Netbeans :
    if input_word[-1]==":":
        input_word_updated=input_word[:-1]
        #input_word_updated=input_word[:-1]+' :'
        ##print("input_word_updated: ", input_word_updated)
        new_token=[input_word_updated,":"]
    return new_token

# JT: 2018-08: Split_On_last_letter_Quote_Mark(input_word) will create a spcace between Quote and letter
# ' Netbeans"'=>'Netbeans "'
def Split_On_last_letter_Quote_Mark(input_word):
    ##print("input_word for Split_On_Non_function_end_parenthesis:", input_word)
    new_token=[input_word]

    if len(input_word)==1:
        return new_token

    #if the word is from a class function do not split end bracket
    class_func_name_rule=re.compile(Class_Func_Name)
    class_func_words = class_func_name_rule.findall(input_word)

    if len(class_func_words)>0:
        return new_token

    #if the word is from a function do not split end bracket
    func_name_rule=re.compile(Func_Name)
    func_words = func_name_rule.findall(input_word)
    if len(func_words)>0:
        return new_token
    if input_word.count("'")==1 and input_word[-1]=="'":
        #input_word_updated=input_word[:-1]+" '"
        #new_token=[input_word_updated]
        input_word_updated=input_word[:-1]
        new_token=[input_word_updated," '"]
    
    if input_word.count('"')==1 and input_word[-1]=='"':
        #input_word_updated=input_word[:-1]+' "'
        #new_token=[input_word_updated]
        input_word_updated=input_word[:-1]
        new_token=[input_word_updated,' "']
    
        
    
    return new_token

# JT: 2018-08: Split_Words_Inside_Parenthesis(input_word) will create a in between spcace between parenthesis words
# '{2,4,5,6,7,8}' => ' { 2 , 4 , 5 , 6 , 7 , 8 } ' 

def Split_Words_Inside_Parenthesis(input_word):
    new_token=[input_word]
    if (input_word[0]=="[" and input_word[-1]=="]") or (input_word[0]=="{"and input_word[-1]=="}") or (input_word[0]=="("and input_word[-1]==")"):
        #print(input_word)
        input_word = input_word.replace(","," , ")
        input_word = input_word.replace("{"," { ")
        input_word = input_word.replace("}"," } ")
        input_word = input_word.replace("["," [ ")
        input_word = input_word.replace("]"," ] ")
        new_token=[input_word]
        #print(input_word)
    return new_token

# JT: 2018-08: Split_Parenthesis_At_End_of_URL(input_word) will split the end of closing parenthesis from the input word if the input word is a url
# 'https://www.sample-videos.com/text/Sample-text-file-10kb.txt)' => 'https://www.sample-videos.com/text/Sample-text-file-10kb.txt )'
# 'https://www.sample-videos.com/text/(Sample-text-file-10kb.txt)' => 'https://www.sample-videos.com/text/(Sample-text-file-10kb.txt)'

def Split_Parenthesis_At_End_of_URL(input_word):
    new_token=[input_word]
    url_rule=re.compile(url)
    url_words = url_rule.findall(input_word)
    word_wo_balanced_paren = []
    for word in url_words:
        bal_paren_word = find_word_w_balanced_paren(word)
        if len(bal_paren_word)==0:
            word_wo_balanced_paren.append(word)
    if len(url_words)>0 and len(word_wo_balanced_paren)>0:
        if input_word[-1]==")" or input_word[-1]=="]" or input_word=="}":
            main_url=input_word[:-1]
            end_paren=")"
            new_token=[main_url,end_paren]
            #print(input_word2)


    #print(url_words)
    return new_token


def SO_Tokenizer_wrapper(tokens):
    #print("input token to SO_TOkenizer_wrapper: ",tokens)
    end_of_sent_punc_split_tokens=Split_End_of_Sentence_Punc(tokens)

    dot_split_tokens=[]
    for token in end_of_sent_punc_split_tokens:
        multiple_dot_splitted_result = Split_On_Multiple_Dot(token)
        if len(multiple_dot_splitted_result)==0:
            dot_split_tokens.append(token)
            continue
        dot_split_tokens.extend(multiple_dot_splitted_result)

    end_parenthesis_split_tokens=[]

    for token in dot_split_tokens:
        end_parenthesis_split_tokens.extend(Split_On_Non_function_end_parenthesis(token))

    end_paren_split_tokens=[]
    for token in end_parenthesis_split_tokens:
        end_paren_split_tokens.extend(Split_On_last_letter_Colon_Mark(token))

    end_of_quote_split_tokens=[]
    for token in end_paren_split_tokens:
        end_of_quote_split_tokens.extend(Split_On_last_letter_Quote_Mark(token))

    inside_parenthesis_split_words=[]
    for token in end_of_quote_split_tokens:
        inside_parenthesis_split_words.extend(Split_Words_Inside_Parenthesis(token))

    

    url_word_end_paren_removed_tokens=[]
    for token in inside_parenthesis_split_words:
        url_word_end_paren_removed_tokens.extend(Split_Parenthesis_At_End_of_URL(token))


    new_tokens=url_word_end_paren_removed_tokens


    multi_space_removed_tokens=[]
    for w in new_tokens:
        #print(w,len(w))
        w =re.sub( '\s+', ' ',w).strip()
        
        #print(w,len(w))
        multi_space_removed_tokens.append(w)


    

   
    new_token=multi_space_removed_tokens
    ##print("new_tokens", new_tokens)

    return new_tokens

def match_paren(current,previous):
    if current==")" and previous=="(":
        return True
    if current=="}" and previous=="{":
        return True
    if current=="]" and previous=="[":
        return True
    return False

def find_word_w_balanced_paren(line):
    list_of_w_balaned_parentheses=[]
    opening_barckets=["(","[","{"]
    closing_barckets=[")","]","}"]
    for word in line.split():
        count_first=word.count("(")
        count_second=word.count("{")
        count_third=word.count("[")
        if count_first+count_second+count_third <=1 and (word[0]=="(" or word[0]=="{" and word[0]=="["): continue
        stack=[]
        paren_found=False
        paren_balanced=False
        if "(" in word and ")" not in word or ")" in word and "(" not in word: 
            continue
        if "{" in word and "}" not in word or "}" in word and "{" not in word: 
            continue
        if "[" in word and "]" not in word or "]" in word and "[" not in word: 
            continue
        for char in word:
            if char in opening_barckets:
                ##print(word,char)
                paren_found=True
                stack.append(char)
            if char in closing_barckets:
                ##print(word,char)
                prev_paren=stack.pop()
                ##print(char,prev_paren)
                paren_balanced=match_paren(char,prev_paren)
                ##print(paren_balanced)
                if paren_balanced==False:
                    break
        if len(stack)==0 and paren_found==True and paren_balanced==True:
            list_of_w_balaned_parentheses.append(word)
            
    return list_of_w_balaned_parentheses


def Mask_Nested_Paren_HTML_Word(text):
    base=""
    counter=0
    Nested_Parenthesis_Words_Dict={}

    for i in range(0,80):
        base+="x"
    ##print(base)
    HTML_Tag_Rule=re.compile(HTML_Tag)
    HTML_Tag_Words = HTML_Tag_Rule.findall(text)

    ##print("HTML_Tag_Words: ",HTML_Tag_Words)

    for w in HTML_Tag_Words:
        counter+=1
        mask_string=base+str(counter)
        Nested_Parenthesis_Words_Dict[mask_string]=w
        text=text.replace(w,mask_string)

    ##print("HTML_masked_text: ",text)
    Neseted_Paren_Words = find_word_w_balanced_paren(text)
    
    ##print("Neseted_Paren_Words from Mask_Nested_Parenthesis_Words: ",Neseted_Paren_Words)
    
    
    modified_text_input=[]
    for word in text.split():
        if word in Neseted_Paren_Words:
            #c#print(word)
            counter+=1
            mask_string=base+str(counter)
            Nested_Parenthesis_Words_Dict[mask_string]=word
            modified_text_input.append(mask_string)
        else:
            modified_text_input.append(word)
    ##print("modified_text_input tokens: ", modified_text_input)
    modified_text_input_str=" ".join(modified_text_input)
    return (Nested_Parenthesis_Words_Dict, modified_text_input_str,base)



def Resotre_Masked_Words(tokens,nested_parenthesis_words_dict, base):
    tokens_unmasked= []
    ##print("input to Resotre_Masked_Words: ", tokens, nested_parenthesis_words_dict)
    for token in tokens:
        ##print("token: ",token)
        if token not in nested_parenthesis_words_dict:
            tokens_unmasked.append(token)
        else:
            main_word=nested_parenthesis_words_dict[token]
            tokens_unmasked.append(main_word)
    tokens_unmasked_wrapper=[]
    for token in tokens_unmasked:
        if base in token:
            split_token=token.split()
            for t in split_token:
                if t in nested_parenthesis_words_dict:
                    t=nested_parenthesis_words_dict[t]
                tokens_unmasked_wrapper.append(t)
        else:
            tokens_unmasked_wrapper.append(token)

            
    ##print(len(tokens_unmaskedc))
    return tokens_unmasked_wrapper


# This is intended for raw tweet text -- we do some HTML entity unescaping before running the tagger.
#
# This function normalizes the input text BEFORE calling the tokenizer.
# So the tokens you get back may not exactly correspond to
# substrings of the original text.
def tokenize(text):
    ##print("main input to tokenizer: ",text)
    (nested_parenthesis_words_dict , nested_paren_masked_text ,base) = Mask_Nested_Paren_HTML_Word(text)
    ##print("nested_paren_masked_text: ", nested_paren_masked_text)
    ##print("nested_parenthesis_words_dict: ", nested_parenthesis_words_dict)
    
    tokens = tokenize_text(normalizeTextForTagger(nested_paren_masked_text))

    tokens_unmasked=Resotre_Masked_Words(tokens,nested_parenthesis_words_dict,base)
    ##print("tokens_unmasked: ", tokens_unmasked)
    tokens_wrapped = SO_Tokenizer_wrapper(tokens_unmasked)

    
    return tokens_wrapped


if __name__ == '__main__':
    # for line in sys.stdin:
    #     #print(' '.join(tokenizeRawTweetText(line)))
    text="In js $http.get().success GNU/Linux 4.2.6-200.fc22.x86_64 you call entityManager.fetchMetadata().then(success, failed) method's first data parameter in templates/your_template/index.php;"
    #text='I do think that the request I send to my API\'s should be more like {post=>{"kind"=>"GGG"}} and not {"kind"=>"GGG"} of I have to find a way for my api to work with the request I send now.'
    #text="I tried to add a manual refresh on the history.state change, I found something like this:"
    #text="In js you call entityManager.fetchMetadata().then(success, failed); After the promise of fetchMetadata() is resolved, breeze metadata of variable entityManager is full-filled and you can start working with your server-side objects in js with breeze!"
    #text="Then in vs. e.g. another file (res.php) I have a sql request and I display results ."
    #text="Modify the img tag like: <'img src=\"<'c:url value='<%=request.getContextPath()%>/images/logo.jpg'> "
    #text="localhost:9200/_search"
    #text="Finally, since O((n^2 , n) 2) = O(n^2) you get that the terms that include the sum dominate the runtime and that is why the algorithm is O(n^2)"
    #text="apache_setenv('sessionID', session_id(), TRUE)"
    #text=":)"
    # #print("input text: ",text)
    # text="Lines 2, 4, and 8 run in O(1) time, so {2,4,5,6,7,8} run in O(1 + 1 + A.Length + 1) = O(A.Length)time."
    # text='Use android:button"@btmImage link"'
    #text="[ERROR] [gwtearthdemo] - Line 96:  O(1) time, so {2,4,5,6,7,8} run in O(1 + 1 + A.Length + 1) = O(A.Length)time."
    #text="So when we write a function like preorderTree :: Tree a -> [a]."
    #text="But now that the div's are vs. floating, they don't push #main's white background down c.net."
    #text="Example: I've confirmed that if I pass a url to some other sample url like (https://www.sample-videos.com/text/Sample-text-file-10kb.txt), my custom webview and google docs will display the results correctly, so the failure only happens when using a url/download going to our custom server."
    #text="Lines 2, 4, and 8 run in O(1) time, so {2,4,5,6,7,8} run in O(1 + 1 + A.Length + 1) = O(A.Length)time."
    tokens = tokenize(text)
    print("output tokens: ",tokens)
    print("output: "," ".join(tokens))

    