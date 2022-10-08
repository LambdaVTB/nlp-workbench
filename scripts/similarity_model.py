from turtle import distance
from scipy import spatial
from sent2vec.vectorizer import Vectorizer



class SimilarityModel:
    def __init__(self, weights: str = "DeepPavlov/rubert-base-cased" ) -> None:
        self.vectorizer = Vectorizer(weights)

    def calc_embeddings(self, sentences: list[str]) -> list:
        # Outputs embeddings for each sentence in the list
        # Efficient for batch processing
        vectors = []
        self.vectorizer.run(sentences)
        vectors = self.vectorizer.vectors[-len(sentences):]
        return self.vectors

    def calc_distance_matrix_from_embeddings(self, embeddings1: list, embeddings2: list) -> list:
        distances = []
        for vector in embeddings1:
            distances.append([])
            for vector2 in embeddings2:
                distances[-1].append(spatial.distance.cosine(vector, vector2))
        return distances


    def calc_distance(self, sentence1: str, sentence2: str) -> float:
        # USE FOR TESTS ONLY
        # IT IS UNIMAGINABLY SLOW IN PRODUCTION
        # USE LATER calc_similarity_batch
        self.vectorizer.run([sentence1, sentence2])
        vectors = self.vectorizer.vectors[-2:]
        return spatial.distance.cosine(vectors[0] / len(sentence1), vectors[1] / len(sentence2))

    def calc_distance_batch(self, sentence1: str, sentences: list[str]) -> list:
        self.vectorizer.run([sentence1] + sentences)
        vectors = self.vectorizer.vectors[-(len(sentences) + 1):]
        distances = []
        for vector in vectors[1:]:
            distances.append(spatial.distance.cosine(vectors[0], vector))
        return distances

    def calc_distance_matrix(self, sentences: list[str]) -> list:
        self.vectorizer.run(sentences)
        vectors = self.vectorizer.vectors[-len(sentences):]
        distances = []
        for vector in vectors:
            distances.append([])
            for vector2 in vectors:
                distances[-1].append(spatial.distance.cosine(vector, vector2))
        return distances


distance_matrix = []

if __name__ == "__main__":
    import pandas as pd

    vectorizer = SimilarityModel("sberbank-ai/sbert_large_nlu_ru")
    # vectorizer = SimilarityModel()
    # vectorizer = SimilarityModel("distilbert-base-uncased")

    sentences = [
        "Рэпер Паша Техник впал в кому после отдыха на вписке с наркотиками",
        "Рэпер Паша Техник впал в кому после вечеринки с наркотиками",
        "Я не люблю кошек",
        "Я не люблю собак",
        "Я боюсь собак",
        "Я боюсь Пашу Техника",
        "Путин отмечает 70-летие",
        "Владимир Путин отмечает юбилей: президенту России исполнилось 70 лет",
        "Владимиру Путину — 70 лет! Топ-5 интересных фактов о жизни президента России",
        "Владимиру Путину исполнилось 70 лет",
        "День рождения Путина: президенту России исполнилось 70 лет",
    ]

    from pprint import pprint
    print("="*80)
    pprint(vectorizer.calc_distance(sentences[0], sentences[1]))
    print("="*80)
    pprint(vectorizer.calc_distance_batch(sentences[0], sentences[1:]))
    print("="*80)
    distance_matrix = vectorizer.calc_distance_matrix(sentences)
    # Convert to pandas df
    df = pd.DataFrame(distance_matrix, columns=sentences, index=sentences)
    # print matrix
    for line in distance_matrix:
        print(line)


    pprint(df < 0.1)
    # import numpy as np
    # answers = [
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0.5, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    # ]
    # answers = - (np.array(answers) - 1)
    # distance_matrix = np.array(distance_matrix)
    # step = 0.1
    # for i in range(0, 1, step):
    #     similar_df = pd.DataFrame(answers, columns=sentences, index=sentences) < i

    #     err = abs(answers - distance_matrix).sum()
    # # Calculate accuracy

    print("="*80)

    # vectors = vectorizer.vectors[-3:]
    # dist_1 = spatial.distance.cosine(vectors[0], vectors[1])
    # dist_2 = spatial.distance.cosine(vectors[0], vectors[2])
    # print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
    # assert dist_1 < dist_2

