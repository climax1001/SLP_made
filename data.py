import nltk
from konlpy.tag import Okt
from konlpy.tag import Kkma
from Korpora import Korpora
okt = Okt()
kkm = Kkma()
text_set = ['나는 밥을 먹는 것을 좋아합니다.',
            '동물원에 사자가 있습니다.',
            '집에 불이 났습니다.']

# for i, line in enumerate(text_set):
#     set_pos = kkm.pos(line)
#     print(set_pos)
#     set_noun = kkm.nouns(line)
#     print(set_noun)

corpus = Korpora.load("modu_spoken")

