import re
import nltk
from nameparser.parser import HumanName
from nltk.corpus import wordnet
import sys
reload(sys)
sys.setdefaultencoding('utf8')

person_list = []
person_names=person_list

def get_human_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)

    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

def srt_time_to_seconds(time):
    split_time=time.split(',')
    major, minor = (split_time[0].split(':'), split_time[1])
    return int(major[0])*1440 + int(major[1])*60 + int(major[2]) + float(minor)/1000

def srt_to_dict(srtText):
    subs=[]
    for s in re.sub('\r\n', '\n', srtText).split('\n\n'):
        st = s.split('\n')
        if len(st)>=3:
            split = st[1].split(' --> ')
            subs.append({'start': srt_time_to_seconds(split[0].strip()),
                         'end': srt_time_to_seconds(split[1].strip()),
                         'text': '<br />'.join(j for j in st[2:len(st)])
                        })
    return subs


with open('input1.srt', "r") as f:
    srtText = f.read()

dictionary = srt_to_dict(srtText)
text = ''
for item in dictionary:
	text = text + ' {}'.format(item['text'])


names = get_human_names(text.encode('utf8'))
for person in person_list:
    person_split = person.split(" ")
    for name in person_split:
        if wordnet.synsets(name):
            if(name in person):
                person_names.remove(person)
                break
for person in person_names:
	print(person)