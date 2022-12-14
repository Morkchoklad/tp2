import csv
import re
import string
# use file object as iterator to read large files

file_name = 'interest.acl94.txt'
sample = 'interest.acl94 sample.txt'
stoplist_file_name = 'stoplist-english.txt'
header = ['m-2','m-1','m1','m2','c-2','c-1','c1','c2','sens']
punctuation = string.punctuation
punctuation = punctuation.replace('%', '')
punctuation = punctuation.replace('/', '')

stoplist = []

with open(stoplist_file_name, 'r') as r:
    for line in r:
        stoplist.append(line.replace('\n',''))

# TODO change to real file, ALSO WINDOWS NEWLINE IS BLANK
with open(file_name, 'r') as r, open('output.csv', 'w', newline='') as w:
    csv_writer = csv.writer(w)
    csv_writer.writerow(header)

    for line in r:  # better alternative to readlines for large files
        # TODO here we suppose each line is a phrase
        phrase = line
        # get rid of brackets
        phrase = phrase.replace(' [ ', ' ')
        phrase = phrase.replace(' ] ', ' ')
        # Convert plural to singular for interest(s), lose underscore in sense labels
        phrase = re.sub('(interests_)|(interest_)', 'interest', phrase)
        # Replace all punctuation within values with X, since WEKA doesn't like punctuation
        # (e.g. u.s. becomes uXsX)
        # Except %, which turns into PERCENT (worth keeping for this problem!)
        # TODO do we replace '.' with X too?
        phrase = phrase.translate({ord(i): 'X' for i in punctuation})
        phrase = phrase.replace('%', 'PERCENT')
        # TODO skipped coarser sense distinctions, optional???
        phrase_list = phrase.split(' ')
        # TODO list of nominal occurences. optional???
        # clean phrase of stoplist words
        phrase_list_copy = phrase_list.copy()
        for pair in phrase_list_copy:
            if pair.split('/')[0] in stoplist:
                phrase_list.remove(pair)

        if phrase_list[0] == 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' or phrase_list[0] == "X":
            phrase_list.insert(1,'X/X')
            phrase_list.insert(1,'X/X')
        else :
            phrase_list.insert(0,'X/X')
            phrase_list.insert(0,'X/X')
        phrase_list.insert(-1,'X/X')
        phrase_list.insert(-1,'X/X')

        phrase_list = list(map(lambda x: x.replace('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'X/X'), phrase_list))
        phrase_list = list(map(lambda x: x.replace('MGMNP', 'X/X'), phrase_list))

        # interest pattern
        interest_pattern = re.compile('(interest\d\/\S*)')
        
        for i, pair in enumerate(phrase_list):
            if interest_pattern.match(pair):
                # get sens
                sens_pattern = re.compile('(\d\/)')
                sens = sens_pattern.search(pair).group(1).split('/')[0]

                #            0      1     2    3     4    5     6    7    8
                # header = ['m-2','m-1','m1','m2','c-2','c-1','c1','c2','sens']
                row = [None,None,None,None,None,None,None,None,sens]
                # check for nulls
                # TODO check for broken phrases like if only has <5 words...
                
                
                # print(phrase_list)
                print()
                
                if i == 0:
                    pair1 = phrase_list[i + 1].split('/')
                    row[2], row[6] = pair1[0], pair1[1]
                    pair2 = phrase_list[i + 2].split('/')
                    row[3], row[7] = pair2[0], pair2[1]
                elif i == 1:
                    pair_1 = phrase_list[i - 1].split('/')
                    row[1], row[5] = pair_1[0], pair_1[1]
                    pair1 = phrase_list[i + 1].split('/')
                    row[2], row[6] = pair1[0], pair1[1]
                    pair2 = phrase_list[i + 2].split('/')
                    row[3], row[7] = pair2[0], pair2[1]
                elif i == len(phrase_list) - 1:
                    pair_1 = phrase_list[i - 1].split('/')
                    row[1], row[5] = pair_1[0], pair_1[1]
                    pair_2 = phrase_list[i - 2].split('/')
                    row[0], row[4] = pair_2[0], pair_2[1]
                elif i == len(phrase_list) - 2:
                    pair_1 = phrase_list[i - 1].split('/')
                    row[1], row[5] = pair_1[0], pair_1[1]
                    pair_2 = phrase_list[i - 2].split('/')
                    row[0], row[4] = pair_2[0], pair_2[1]
                    pair1 = phrase_list[i + 1].split('/')
                    row[2], row[6] = pair1[0], pair1[1]
                else:
                    
                    
                    pair_1 = phrase_list[i - 1].split('/')
                    row[1], row[5] = pair_1[0], pair_1[1]
                    pair_2 = phrase_list[i - 2].split('/')
                    
                        
                    
                    
                    row[0], row[4] = pair_2[0], pair_2[1]
                    pair1 = phrase_list[i + 1].split('/')
                    print(phrase_list)
                    print(pair1)
                    row[2], row[6] = pair1[0], pair1[1]
                    pair2 = phrase_list[i + 2].split('/')
                    row[3], row[7] = pair2[0], pair2[1]
                
                csv_writer.writerow(row)
                





