import re

# items="a|b => regex_or(items) ==> (?:a|b)
def regex_or(*items):
    return '(?:' + '|'.join(items) + ')'
Bound = r"(?:\W|^|$)"

punctChars = r"['\"“”‘’.?!…,:;]"
# punctSeq   = punctChars+"+"  #'anthem'. => ' anthem '.
punctSeq = r"['\"“”‘’]+|[.?!,…]+|[:;]+"  # 'anthem'. => ' anthem ' .
entity = r"&(?:amp|lt|gt|quot);"  # html tag entities &quot; => "


urlStart1 = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx|aspx)"
ccTLDs = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
         r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
         r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
         r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
         r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
         r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
         r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
         r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"  # TODO: remove obscure country domains?
urlStart2 = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs,
                                                                      ccTLDs) + r"(?:\." + ccTLDs + r")?(?=\W|$)"
urlBody = r"(?:[^\.\s<>][^\s<>]*?)?"
urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
urlEnd = r"(?:\.\.+|[<>]|\s|$)"
url = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:" + urlExtraCrapBeforeEnd + ")?" + urlEnd + ")"
url_rule=re.compile(url)


# JT: 2018-08
# '.html.erb' => '. html . erb' :: WRONG
# '.html.erb' => '.html.erb' :: WRONG
File_Ext = r"[.]?[\w.\-]*\.[\w]+(?=" + Bound + ")"

# JT: 2018-08
Path = r'(?:/?[\w\-.]+\/+)+'
# Path=r'(?:/?[\w\-.]+\/+[\w\-.]*)+'

# JT: 2018-08
# '/templates/your_template/.php.ini' => '/templates/your_template/ . php . ini' :: WRONG
# '/templates/your_template/.php.ini' => '/templates/your_template/.php.ini' :: RIGHT
File_Path_W_File_Name = Path + "(?:" + File_Ext + ")*"
file_rule = re.compile(File_Path_W_File_Name)

def IS_URL(token):
	list_test=url_rule.findall(token)
	if len(list_test)>0:
		return True
	return False


def IS_NUMBER(token):
	token_updated=token.replace(".","").replace("-","").replace("+","")
	if token_updated.isdigit():
		return True
	return False


def IS_FILE_NAME(token):
	list_test=file_rule.findall(token)
	if len(list_test)>0:
		return True
	return False

if __name__ == '__main__':
	
	str_test= "-12.4a"
	
	# print(IF_URL(str_test))
	print(IS_NUMBER(str_test))
