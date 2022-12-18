import csv
import re
import string
# use file object as iterator to read large files

class WindowFeatures:

    def CleanFileToWindows(self, file_name, window_size = 2):
        #file_name = 'interest.acl94.txt'
        sample = 'interest.acl94 sample.txt'
        stoplist_file_name = 'stoplist-english.txt'

        # eg. if window_size is 2 :
        #            0      1     2    3     4    5     6    7    8
        # header = ['m-2','m-1','m1','m2','c-2','c-1','c1','c2','sens']
        header = []
        for i in range(-window_size, window_size + 1, 1):
            if i != 0:
                header.append('m' + str(i))
        for i in range(-window_size, window_size + 1, 1):
            if i != 0:
                header.append('c' + str(i))
        header.append('sens')

        punctuation = string.punctuation
        punctuation = punctuation.replace('%', '')
        punctuation = punctuation.replace('/', '')

        stoplist = []

        with open(stoplist_file_name, 'r') as r:
            for line in r:
                stoplist.append(line.replace('\n', ''))

        # TODO change to real file, ALSO WINDOWS NEWLINE IS BLANK
        with open(file_name, 'r') as r, open('output' + str(window_size) + '.csv', 'w', newline='') as w:
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
                    for _ in range(window_size):
                        phrase_list.insert(1, 'X/X')

                else:
                    for _ in range(window_size):
                        phrase_list.insert(0, 'X/X')

                for _ in range(window_size):
                    phrase_list.insert(-1, 'X/X')

                phrase_list = list(
                    map(lambda x: x.replace('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', 'X/X'), phrase_list))
                phrase_list = list(map(lambda x: x.replace('MGMNP', 'X/X'), phrase_list))

                # interest pattern
                interest_pattern = re.compile('(interest\d\/\S*)')

                for i, pair in enumerate(phrase_list):
                    if interest_pattern.match(pair):
                        # get sens
                        sens_pattern = re.compile('(\d\/)')
                        sens = sens_pattern.search(pair).group(1).split('/')[0]

                        # eg. if window_size is 2 :
                        #          0    1    2    3    4    5    6    7    8
                        # row = [None,None,None,None,None,None,None,None,sens]
                        row = []
                        for _ in range(4*window_size):
                            row.append(None)
                        row.append(sens)

                        # Beginning of sentences
                        if i < window_size:
                            for j in range(i):
                                pair = phrase_list[j].split('/')
                                row[j], row[j + 2 * window_size] = pair[0], pair[1]
                            for j in range(window_size):
                                pair = phrase_list[i + j + 1].split('/')
                                row[window_size + j], row[3 * window_size + j] = pair[0], pair[1]
                        # End of sentences
                        elif len(phrase_list) - i <= window_size:
                            for j in range(window_size):
                                pair = phrase_list[i - window_size + j].split('/')
                                row[j], row[j + 2 * window_size] = pair[0], pair[1]
                            for j in range(len(phrase_list) - i - 1):
                                pair = phrase_list[i + j + 1].split('/')
                                row[window_size + j], row[3 * window_size + j] = pair[0], pair[1]
                        # Base case
                        else:
                            for j in range(window_size):
                                pair = phrase_list[i - window_size + j].split('/')
                                row[j], row[j + 2 * window_size] = pair[0], pair[1]
                            for j in range(window_size):
                                pair = phrase_list[i + j + 1].split('/')
                                row[window_size + j], row[3 * window_size + j] = pair[0], pair[1]

                        csv_writer.writerow(row)

    def GetFeaturesWindow(self, file_name, window_size):
        self.CleanFileToWindows(file_name, window_size)
        file = open('output' + str(window_size) + '.csv')
        csvreader = csv.reader(file)

        header = next(csvreader)
        pos_window = []
        X = []
        for row in csvreader:
            X.append(row)
        y = []
        for row in X:
            rowDict = {}
            for i in range(4*window_size):
                rowDict.update({header[i] : row[i]})
            y.append(row[-1])
            pos_window.append(rowDict)

        return pos_window, y

    def CleanFileToSentences(self, file_name):
        #file_name = 'interest.acl94.txt'
        sample = 'interest.acl94 sample.txt'
        stoplist_file_name = 'stoplist-english.txt'

        punctuation = string.punctuation
        punctuation = punctuation.replace('%', '')
        punctuation = punctuation.replace('/', '')

        stoplist = []

        with open(stoplist_file_name, 'r') as r:
            for line in r:
                stoplist.append(line.replace('\n', ''))

        # TODO change to real file, ALSO WINDOWS NEWLINE IS BLANK
        with open(file_name, 'r') as r, open('outputSentences.csv', 'w', newline='') as w:
            csv_writer = csv.writer(w)

            for line in r:  # better alternative to readlines for large files
                # TODO here we suppose each line is a phrase
                phrase = line
                # get rid of unwanted characters
                phrase = phrase.replace(' [ ', ' ')
                phrase = phrase.replace(' ] ', ' ')
                phrase = phrase.replace('$$\n', '')
                phrase = phrase.replace('====================================== ', '')

                # if the string is empty, skip to next line
                if phrase == '':
                    continue

                # Convert plural to singular for interest(s), lose underscore in sense labels
                phrase = re.sub('(interests_)|(interest_)', 'interest', phrase)

                # Replace all punctuation within values with X, since WEKA doesn't like punctuation
                # (e.g. u.s. becomes uXsX)
                # Except %, which turns into PERCENT (worth keeping for this problem!)
                phrase = phrase.translate({ord(i): 'X' for i in punctuation})
                phrase = phrase.replace('%', 'PERCENT')

                phrase_list = phrase.split(' ')

                # clean phrase of stoplist words
                phrase_list_copy = phrase_list.copy()
                for pair in phrase_list_copy:
                    if pair.split('/')[0] in stoplist:
                        phrase_list.remove(pair)

                phrase_list = list(map(lambda x: x.replace('MGMNP', 'X/X'), phrase_list))

                while phrase_list.count('\n') != 0:
                    phrase_list.remove('\n')
                while phrase_list.count('X') != 0:
                    phrase_list.remove('X')
                # interest pattern
                interest_pattern = re.compile('(interest\d\/\S*)')
                phrase_length = len(phrase_list) - 1
                next_index = 0
                row = []
                for _ in range(2 * phrase_length + 1):
                    row.append(None)
                for i, pair in enumerate(phrase_list):
                    if interest_pattern.match(pair):
                        # get sens
                        sens_pattern = re.compile('(\d\/)')
                        sens = sens_pattern.search(pair).group(1).split('/')[0]
                        # eg. if phrase_length is 4 :
                        #          0    1    2    3    4    5    6    7    8
                        # row = [None,None,None,None,None,None,None,None,sens]
                        row[2 * phrase_length] = sens

                    else :
                        pair = phrase_list[i].split('/')
                        row[next_index], row[phrase_length + next_index] = pair[0], pair[1]
                        next_index += 1
                csv_writer.writerow(row)

    def GetFeaturesSentences(self, file_name):
        self.CleanFileToSentences(file_name)
        file = open('outputSentences.csv')
        csvreader = csv.reader(file)

        pos_sentence = []
        X = []
        for row in csvreader:
            X.append(row)
        y = []
        for row in X:
            y.append(row.pop())
            rowDict = {}
            for i in range(len(row)//2):
                rowDict.update({'m' + str(i) : row[i]})
            for i in range(len(row)//2):
                rowDict.update({'c' + str(i) : row[i + len(row)//2]})

            pos_sentence.append(rowDict)

        return pos_sentence, y