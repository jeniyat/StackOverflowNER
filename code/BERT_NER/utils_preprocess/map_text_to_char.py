import stokenizer #JT: Dec 6


def map_text_to_char(main_sent, tokens, offset):
	tokenized_sent=" ".join(tokens)
	# print("tokenized_sent: ", tokenized_sent)

	main_sent_iter=0
	tokenized_sent_iter=0
	tokenized_sent_actual_indices=[]
	while(tokenized_sent_iter<len(tokenized_sent)):
		if tokenized_sent_iter< len(tokenized_sent):
			tokenized_sent_char=tokenized_sent[tokenized_sent_iter]
		if main_sent_iter< len(main_sent):
			main_sent_char=main_sent[main_sent_iter]

		while main_sent_char!=tokenized_sent_char and main_sent_char==" ":
			if main_sent_iter+1==len(main_sent):break
			main_sent_iter+=1
			main_sent_char=main_sent[main_sent_iter]

		while main_sent_char!=tokenized_sent_char and tokenized_sent_char==" ":
			if tokenized_sent_iter+1==len(tokenized_sent):break
			tokenized_sent_iter+=1
			tokenized_sent_char=tokenized_sent[tokenized_sent_iter]
		if tokenized_sent_char!=" ":
			tokenized_sent_actual_indices.append((tokenized_sent_char, main_sent_iter))
		main_sent_iter+=1
		tokenized_sent_iter+=1




		# print(main_sent_char, tokenized_sent_char)
	# print(tokenized_sent_actual_indices)
	word=""
	word_start_at=0
	index=0
	token_iter=0
	token_start_pos=[]
	for t in tokens:

		# print("len(tokenized_sent_actual_indices: ",len(tokenized_sent_actual_indices), token_iter)
		# print(t, tokenized_sent_actual_indices[token_iter][1])
		t1=t.replace("-----"," ")
		if token_iter<len(tokenized_sent_actual_indices):
			token_start_pos.append((t, tokenized_sent_actual_indices[token_iter][1]+offset))
		token_iter+=len(t1)
		
	


	return token_start_pos



if __name__ == '__main__':
	# text="TextView       has setText(String), but when looking on the Doc, I don't see one for GridLayout."
	text="   NetBeans: How to use .jar files in NetBeans want(With-----a-----maximum-----size-----of-----4608x3456)?"
	print("main text: ",text)
	tokens = stokenizer.tokenize(text)
	print("split_token: ", tokens)
	token_W_pos = map_text_to_char(text, tokens, 0)
	print("token_W_pos: ", token_W_pos)

