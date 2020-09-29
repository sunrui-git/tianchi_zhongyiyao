import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from tokenizer import segment
# from seq2seq_tf2.bin.main import BASE_DIR
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']


# 获取停用词
def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


# 删除REMOVE_WORDS中的字符
def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def parse_data(path):
    data = pd.read_json(path, encoding='utf-8')
    texts = data['text']
    questions = []
    answers = []
    annotations = data['annotations']
    for annotation in annotations:
        for line in annotation:
            questions.append(line['Q'])
            answers.append(line['A'])
    return texts, questions, answers


def save_data(data_1, data_2, data_3, data_path_1, data_path_2, data_path_3, stop_words_path=''):
    stopwords = read_stopwords(stop_words_path)
    with open(data_path_1, 'w', encoding='utf-8') as f1:
        count_1 = 0
        for line in data_1:
            # print(line)
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                # seg_words = []
                # for j in seg_list:
                #     if j in stopwords:
                #         continue
                #     seg_words.append(j)
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f1.write('%s' % seg_line)
                    f1.write('\n')
                    count_1 += 1
        print('train_x_length is ', count_1)

    with open(data_path_2, 'w', encoding='utf-8') as f2:
        count_2 = 0
        for line in data_2:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                # seg_words = []
                # for j in seg_list:
                #     if j in stopwords:
                #         continue
                #     seg_words.append(j)
                # if len(seg_list) > 0:
                seg_line = ' '.join(seg_list)
                f2.write('%s' % seg_line)
                f2.write('\n')
                count_2 += 1
        print('train_y_length is ', count_2)

    with open(data_path_3, 'w', encoding='utf-8') as f3:
        count_3 = 0
        for line in data_3:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f3.write('%s' % seg_line)
                    f3.write('\n')
                    count_3 += 1
        print('test_y_length is ', count_3)


def preprocess_sentence(sentence):
    # 分词
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    # 需要更换成自己数据的存储地址
    train_texts, trian_questions, train_answers = parse_data('{}/datasets/round1_train_0907.json'.format(BASE_DIR))
    test_texts, _, test_answers = parse_data('{}/datasets/round1_test_0907.json'.format(BASE_DIR))
    train_texts.apply(preprocess_sentence).to_csv('{}/datasets/train_texts.txt'.format(BASE_DIR), index=None, header=False)
    pd.Series(trian_questions).apply(preprocess_sentence).to_csv('{}/datasets/train_questions.txt'.format(BASE_DIR), index=None, header=False)
    pd.Series(train_answers).apply(preprocess_sentence).to_csv('{}/datasets/train_answers.txt'.format(BASE_DIR), index=None, header=False)
    pd.Series(test_texts).apply(preprocess_sentence).to_csv('{}/datasets/test_texts.txt'.format(BASE_DIR), index=None, header=False)
    pd.Series(test_answers).apply(preprocess_sentence).to_csv('{}/datasets/test_answers.txt'.format(BASE_DIR), index=None, header=False)

    # save_data(train_list_src,
    #           train_list_trg,
    #           test_list_src,
    #           '{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
    #           '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
    #           '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
    #           stop_words_path='{}/datasets/stop_words.txt'.format(BASE_DIR))


