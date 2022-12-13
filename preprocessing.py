import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from tqdm import tqdm

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data.drop_duplicates(subset=['text'], inplace=True)
train_data['text'] = test_data['text'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
train_data['text'] = test_data['text'].str.replace(
    '^ +', "")  # 공백은 empty 값으로 변경
train_data['text'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
train_data = train_data.dropna(how='any')

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
             '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
