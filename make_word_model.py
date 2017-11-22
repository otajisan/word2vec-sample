import os
import glob
import MeCab
mecab = MeCab.Tagger('mecabrc')

from gensim.models import word2vec

DATASET_PATH = './dataset'
MODLE_PATH = './models'

def tokenize(sentence):
    '''
    文章を単語に分かち書きする
    '''
    white_list = ["名詞", "動詞", "形容詞", "形容動詞", "助動詞"]
    node = mecab.parseToNode(sentence)
    word_list = []
    while node:
        # 不要な品詞をフィルタリング
        feature = node.feature.split(',')
        speech = feature[0] # 品詞
        word = feature[6] # 単語
        if speech in white_list:
            word_list.append(word)

        node = node.next
    return word_list

def tokenize_category(base_path, category):
    # 前回ファイルを削除
    out_file_name = os.path.join(DATASET_PATH, category + '.txt')
    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    # カテゴリ内のファイルを走査
    category_dir = os.path.join(base_path, category)
    file_list = os.listdir(category_dir)
    for file in file_list:
        for sentence in open(os.path.join(category_dir, file), 'r', encoding='utf-8'):
            # 文章 -> 単語変換
            words = tokenize(sentence)
            # 結果を書き込み
            out_file = open(out_file_name, 'a')
            out_file.write(" ".join(words) + " ")

def tokenize_all_categories(path):
    category_dirs = os.listdir(path)
    # すべてのカテゴリを走査
    for category in category_dirs:
        if '.txt' in category:
            continue
        # 1カテゴリ毎に文章を単語化
        tokenize_category(path, category)

def vectorize(path):
    word_files = glob.glob(DATASET_PATH + '/*.txt')
    category = 'hoge'
    for word_file in word_files:
        # 形態素解析したデータを読み込み
        sentences = word2vec.Text8Corpus(word_file)
        # ハイパーパラメータ
        sg = 1
        size = 300
        min_count = 10
        window = 5
        hs = 0
        negative = 15
        iter = 15
        # モデル生成
        model = word2vec.Word2Vec(sentences, sg=sg, size=size, min_count=min_count, window=window, hs=hs, negative=negative, iter=iter)
        # モデルをファイルに保存
        model.save(os.path.join(MODLE_PATH, category))

if __name__ == '__main__':
    # 文章 -> 単語に変換し、ファイルに保存
    tokenize_all_categories(DATASET_PATH)
    # 単語 -> ベクトル変換
    vectorize(MODLE_PATH)
