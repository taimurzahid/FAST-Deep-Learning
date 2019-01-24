from nltk import word_tokenize, pos_tag, ne_chunk, tree

# with open('sample.txt', 'r') as f:
#     txtfile = f.read()
chunks_list = []
f = open('transcript.txt')
for line in f:

    tokens = word_tokenize(line)
    pos_tagged_tokens = pos_tag(tokens)
    ne_chunks = ne_chunk(pos_tagged_tokens)


    
    for subtree in ne_chunks:
        if type(subtree) == tree.Tree: # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            if(ne_label == "PERSON"):
                chunks_list.append((tokens[0],ne_string))
print chunks_list

thefile = open('output_names.txt', 'w')
for item in chunks_list:
  print>>thefile, item[0], item[1]

    # tokens = word_tokenize(txtfile)
    # pos_tagged_tokens = pos_tag(tokens)
    # ne_chunks = ne_chunk(pos_tagged_tokens)


    # chunks_list = []
    # for subtree in ne_chunks:
    #     if type(subtree) == tree.Tree: # If subtree is a noun chunk, i.e. NE != "O"
    #         ne_label = subtree.label()
    #         ne_string = " ".join([token for token, pos in subtree.leaves()])
    #         if(ne_label == "PERSON"):
    #             chunks_list.append(ne_string)
    # print chunks_list