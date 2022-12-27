#%%
import numpy as np
import pandas as pd


from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from surprise import SVD
from surprise import accuracy
from surprise import Reader

#%%
#안 본 영화 id 추출
def getUnseenSurprise(df, userId):
    seenMovies = df[df["userId"]==userId]["movieId"].tolist() #본 영화
    allMovies = movies["movieId"].tolist() #전체 영화
    unseenMovies = [movie for movie in allMovies if movie not in seenMovies] #안 본 영화
    print(f"평점을 매긴 영화 수: {len(seenMovies)}, \n전체 영화 수: {len(allMovies)}, \n추천해야 할 영화 수: {len(unseenMovies)}")
    return unseenMovies

# 예측, predictions 정렬, TopN개의 predictions 추출, TopN개의 id, 제목, 예측평점 데이터프레임 생성
def surpriseRecMovie(algo, userId, unseenMovies, topN):
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseenMovies]
    # 알고리즘 객체의 predidct() 매서드를 평점이 없는 영화에 반복 수행한 후 그 결과를 list 객체로 저장

    def sort_est(pred):
        return pred.est

    predictions.sort(key=sort_est, reverse=True)

    topNpred = predictions[:topN]
    print(topNpred)

    #top10으로 추출된 영화의 정보 추출(movieId, est, title)
    topMovieIds = [int(pred.iid) for pred in topNpred]
    topMovieEsts = [pred.est for pred in topNpred]
    topMovieTitles = movies[movies["movieId"].isin(topMovieIds)]['title']

    topMoviePreds = [(id, title, est) for id, title, est in zip(topMovieIds, topMovieTitles, topMovieEsts)]

    topMovieDf = pd.DataFrame(topMoviePreds, columns=["movieId", "title", "pred"])
    return topMovieDf

# %%
###DatasetAutoFolds
#trainset 정의
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0.5, 5))
daf = DatasetAutoFolds(ratings_file = r'data/ratings_noh', reader=reader)
trainset = daf.build_full_trainset()

#알고리즘 정의
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

###testest 정의
movies = pd.read_csv(r"data/grouplens_movies.csv")
ratings = pd.read_csv(r"data/grouples_ratings.csv")
print(f"movies shape: {movies.shape}")
print(f"ratings shape: {ratings.shape}")

###9번 사용자에게 영화를 추천
#What? 9번 사용자가 평점을 매기지 않은 영화 찾기

# 9번 사용자가 보지 않은 영화들의 id
unseenMovies = getUnseenSurprise(ratings, 9) #9번 사용자의 안 본 영화들의 id
unseenMovies

# 9번 사용자
surpriseRecMovie(algo, 9, unseenMovies, topN=10)
# %%