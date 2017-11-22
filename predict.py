import os
from gensim.models import word2vec

MODEL_PATH = './models'

def predict(positive):
    models = os.listdir(MODEL_PATH)
    for model_file in models:
        if model_file == '.gitkeep':
            continue
        model = word2vec.Word2Vec.load(os.path.join(MODEL_PATH, model_file))
        result = model.most_similar(positive=positive)
        for pair in result:
            word = pair[0]
            distance = pair[1] # コサイン距離
            print(word, distance)

if __name__ == '__main__':
    # 五反田に近いワードを探る
    positive = '五反田'
    predict(positive=positive)
