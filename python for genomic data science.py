#!/usr/bin/env python
# coding: utf-8

# In[2]:


dna = input("Enter a DNA sequence, please: ")


# In[3]:


dna


# In[4]:


my_number = input("Please enter a number: ")


# In[5]:


type(my_number)


# In[6]:


actual_number = int(my_number)


# In[7]:


type(actual_number)


# In[11]:


float(my_number)
type(my_number)


# In[12]:


complex(my_number)
type(my_number)


# In[13]:


chr(65)


# In[14]:


str(65)


# In[15]:


# print("The DNA Sequence's GC content is", gc_perc, "%")

# read DNA sequence from user
# count the number of C's in DNA sequence
# count the number of G's in DNA sequence
# determine the length of the DNA sequence
# compute the GC%
# print GC%


# In[16]:


12/5


# In[17]:


float(12)/5


# In[18]:


type(3+2j)


# In[19]:


print("This is a codon, isn't it?")


# In[20]:


print("This is a codon, isn\'t it?")


# In[22]:


print("""
hi
nice
to
meet
you
""")
print(1+2)


# In[23]:


'agt' +'gtacgtccgt'


# In[24]:


'agt' * 3


# In[25]:


'atg' in 'atggccggcgta'


# In[26]:


'gtacgtccgt'[:3]


# In[27]:


help(len)


# In[28]:


dna = 'aagtccgcgcgctttttaaggagccttttgacggc'


# In[29]:


dna.count('c')


# In[30]:


dna.count('gc')


# In[31]:


dna.upper()


# In[32]:


dna.lower()


# In[33]:


dna.find('ag') #appear in what position?


# In[35]:


dna.find('ag', 2)


# In[36]:


dna.rfind('ag')


# In[37]:


dna.replace('a', 'A')


# In[50]:


"""
This is my first Python program.
It computes the GC content of a DNA sequence.
"""

dna = 'acgctcgcgcggcgatagctgatcgatcggcgcgctttttttttaaaag'
no_c = dna.count('c')
no_g = dna.count('g')
dna_length = len(dna)
gc_percent = (no_c + no_g)*100/dna_length
print("The DNA Sequence's GC content is", gc_percent, "%")


# In[40]:


no_c = dna.count('c')
print(no_c)


# In[41]:


no_g = dna.count('g')
print(no_g)


# In[44]:


dna_length = len(dna)
print(dna_length)


# In[48]:


gc_percent = (no_c + no_g)*100/dna_length
print(gc_percent)


# In[47]:


print("The DNA Sequence's GC content is", gc_percent, "%")


# In[51]:


print("The DNA Sequence's GC content is %5.3f %%" % gc_percent)


# In[52]:


gene_expression = ['Lif', '5.16e-08', '0.000138511', '7.33e-08']


# In[53]:


gene_expression[0]


# In[55]:


gene_expression[-3:]


# In[56]:


gene_expression[1:3] = [6.09e-07]


# In[57]:


gene_expression


# In[58]:


gene_expression[:] = []


# In[59]:


gene_expression


# In[60]:


gene_expression = ['Lif', 6.09e-07, '7.33e-08']


# In[61]:


len(gene_expression)


# In[62]:


del gene_expression[1]


# In[63]:


gene_expression


# In[64]:


gene_expression.extend([5.16e-08, 0.000138511])


# In[65]:


gene_expression


# In[67]:


print(gene_expression.count('Lif'), gene_expression.count('gene'))


# In[71]:


gene_expression.reverse()


# In[72]:


gene_expression


# In[73]:


stack = ['a', 'b', 'c', 'd']


# In[74]:


stack.append('e')


# In[75]:


stack


# In[76]:


elem = stack.pop()


# In[77]:


elem


# In[78]:


mylist = [1, 3, 5, 31, 123]


# In[79]:


sorted(mylist)


# In[80]:


mylist


# In[81]:


mylist.sort()


# In[82]:


mylist


# In[83]:


mylist = ['c', 'g', 'T', 'a', 'A']


# In[84]:


sorted(mylist)


# In[85]:


a = 1,2,3
b = (1,2,3)


# In[86]:


a == b


# In[87]:


brca1 = {'a', 'b', 'c'}


# In[91]:


brca2 = {'c', 'd', 'e', 'f'}


# In[92]:


brca1 | brca2 #union


# In[93]:


brca1 & brca2


# In[94]:


brca1 - brca2


# In[95]:


TF_motif = {
    'SP1': 'gggcgg',
    'C/EBP': 'attgcgcaat',
    'ATF': 'tgacgtca',
    'c-Myc': 'cacgtg',
    'Oct-1': 'atgcaaat'
}


# In[96]:


print("The recognition sequecne for the ATF transcription is %s." % TF_motif['ATF']) 


# In[97]:


TF_motif['AR-1'] = 'tgagtca' #add new on to the dict.


# In[98]:


TF_motif


# In[99]:


del TF_motif['SP1'] #del from the dict.


# In[100]:


TF_motif


# In[101]:


len(TF_motif)


# In[102]:


list(TF_motif.keys())


# In[103]:


list(TF_motif.values())


# In[104]:


sorted(TF_motif.keys())


# f = open('myfile', 'r') <- reading from a file
# open(filename, code)
# 
# f = open('myfile', 'w') <- writing into a file
# ** to append to the end of the file, use mode 'a':
# f = open('myfile', 'a')

# In[106]:


try:
    f = open("file")
except IOError:
    print("the file myfile does not exist!!")


# fasta file : a format for how represet DNA sequences.
#             & name of a classic program for doing sequence alignment
#             one or more DNA sequences
#             
# open file
# read file
# 
# is line a header?
# 
# 'yes' > get sequence name
# create new entry in dictionary
# 
# 'no' > update sequence in dictionary

# In[ ]:


# open & read file
try:
    f = open("myfile.fa")
except IOError:
    print("File myfile.fa does not exist!!")


# In[ ]:


# Reading a FASTA file
seqs = {}
for line in f:
    line = line.rstrip()
    
    if line[0]=='>'
    words = line.split()
    name = words[0][1:]
    seqs[name] = ''
    
    else:
        seqs[name] = seqs[name] + line


# In[ ]:


# Retrieving data from dict.

for name,seq in seq.items():
    print(name,seq)


# In[ ]:


import sys
filename=sys.argv[1]

try:
    f = open(filename)
except IOError:
    print("File %s does not exist!!" % filename)


# In[108]:


import Bio
print(Bio.__version__)


# In[109]:


print("I have %s" % "apple")


# In[110]:


from Bio.Blast import NCBIWWW


# In[111]:


fasta_string = open("myseq.fa").read()
result_handle = NCBIWWW.qblast("blastn", "nt", fasta_string) #nt->which database, blastn->method to use


# In[2]:


#qblast(program, databbase, sequence)


# In[3]:


from Bio.Blast import NCBIXML
blast_record = NCBIXML.read(result_handle)


# In[ ]:




