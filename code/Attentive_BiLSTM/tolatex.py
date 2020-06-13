def tolatex(table_dict,caption=""):
	print("\n\n\n")
	print("\\begin{table}[htbp]")
	print("\\centering")
	print("\\begin{tabular}{|",end='')
	for i in range(len(table_dict["header"])):
		print("c|",end='')
	print("}")
	header=" & ".join(table_dict["header"])+"\\\\"
	print("\\hline")
	print(header)
	print("\\hline")
	for row in table_dict["rows"]:
		row_=[str(x).replace("_"," ").replace("%","\\%") for x in row]
		row_str=" & ".join(row_)+"\\\\"
		print(row_str)
		# print("\\hline")
	print("\\hline")
	print("\\end{tabular}")
	print("\\caption{"+caption+"}")
	#print("\\label{tab:-----}")
	print("\\end{table}")

	print("\n\n\n")



if __name__ == '__main__':
	table_dict={}
	table_dict["header"]=['','Precision', 'Recall', 'F1']
	table_dict["rows"]=[['Algorithm','100.00', '100.00', '100.00'], ['Algorithm','100.00', '100.00', '100.00']]
	tolatex(table_dict)