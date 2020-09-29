from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_texts_path, train_questions_path, train_answers_path,test_texts_path,test_answers_path):
    ret = []
    lines = read_lines(train_texts_path)
    lines += read_lines(train_questions_path)
    lines += read_lines(train_answers_path)
    lines += read_lines(test_texts_path)
    lines += read_lines(test_answers_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build(train_texts_path, train_questions_path, train_answers_path,test_texts_path,test_answers_path
          , out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(train_texts_path, train_questions_path, train_answers_path,test_texts_path,test_answers_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    """
        通过gensim工具完成word2vec的训练，输入格式采用sentences，使用skip-gram，embedding维度256
    """
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('{}/datasets/train_texts.txt'.format(BASE_DIR),
          '{}/datasets/train_questions.txt'.format(BASE_DIR),
          '{}/datasets/train_answers.txt'.format(BASE_DIR),
          '{}/datasets/test_texts.txt'.format(BASE_DIR),
          '{}/datasets/test_answers.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))

