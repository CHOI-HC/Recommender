#%%
import pandas as pd
import numpy as np

from dataprep.eda import plot, create_report

from ast import literal_eval

from sklearn.metrics.pairwise import cosine_similarity
import warnings; warnings.filterwarnings('ignore')

# %%
def weight_average(pct, df):
    pct = pct
    m = df['vote_count'].quantile(0.6)
    c = df['vote_average'].mean()
    v = df['vote_count']
    r = df['vote_average']
    weight_average = (v/(v+m))*r + (m/(v+m))*c
    return weight_average

def weight_vote_average(df, sorted_sim, title_name, num=10):
    title_mv2 = df[df['title']==title_name]
    title_mv2_idx = title_mv2.index.values #해당영화 본 사람
    sim_idx2 = sorted_sim[title_mv2_idx, :(num*2)] #유사도가 높은 10인의 인덱스
    sim_idx2 = sim_idx2.reshape(-1) #1차원으로 변경
    sim_idx2 = sim_idx2[sim_idx2!=title_mv2_idx]
    sim_mv2 = df.iloc[sim_idx2]
    return sim_mv2.sort_values(by='weight_average', ascending=False)[:10]




movies = pd.read_csv(r'data/tmdb_5000_movies.csv')
# create_report(movies)
print(f'movies_shape: {movies.shape}')

###필요한 컬럼 추출
movies_df = movies[['id', 'title', 'genres', 'vote_average', 'vote_count', 'popularity', 'keywords', 'overview']]
print(f'movies_df_shape: {movies_df.shape}')
# movies_df.head(3)

### 장르컬럼 원핫 인코딩
#문자열을 리스트. 딕셔너리 형태로 변환
movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)

#genres 컬럼에서 name만 추출
movies_df['genres'] = movies_df['genres'].apply(lambda x: [y['name'] for y in x])

#장르 유니크 리스트 만들기
genre_list = []
for gen in movies_df['genres']:
    genre_list.extend(gen)
print(f'genre_list len: {len(genre_list)}')
genre_list = np.unique(genre_list)
print(f'unique genre_list len: {len(genre_list)}')

#원핫 인코딩 매트릭스 만들기
zeroMat = np.zeros(shape=(movies_df.shape[0], len(genre_list)))
print(f'zeroMat shape: {zeroMat.shape}')
zero_df = pd.DataFrame(zeroMat, columns=genre_list)

for idx, genre in enumerate(movies_df['genres']):
    indices = zero_df.columns.get_indexer(genre)
    zero_df.iloc[idx, indices] = 1

### 유사도 구하기
gen_df = zero_df.copy()
gen_sim = cosine_similarity(gen_df, gen_df)
print(f'gen_sim shape: {gen_sim.shape}')

#정렬
sorted_gen_sim = gen_sim.argsort()[:,::-1]
print(sorted_gen_sim[0])

#1차 추천
# title_mv = movies_df[movies_df['title']=='The Godfather']
# title_mv_index = title_mv.index.values #해당영화 본 사람

# sim_idx = sorted_gen_sim[title_mv_index, :10] #유사도가 높은 10인의 인덱스
# sim_idx = sim_idx.reshape(-1) #1차원으로 변경

# sim_mv = movies_df.iloc[sim_idx][['title', 'vote_average']]
# print(sim_mv)

##가중평점으로 추천
# 가중평점 = (v/(v+m))*r + (m/(v+m))*c
# v: 영화별 평점 투표 횟수
# m: 평점 부여를 위한 최소 투표 횟수
# r: 영화별 평균 평점
# c: 전체 영화의 평균 평점


movies_df['weight_average'] = weight_average(0.6,movies_df)
# movies_df[['title', 'vote_average', 'weight_average', 'vote_count']].sort_values(by='weight_average', ascending=False)


sim_mv2 = weight_vote_average(movies_df, sorted_gen_sim, 'The Godfather')
sim_mv2 = sim_mv2[['title', 'weight_average', 'vote_average']]
sim_mv2
# print(sim_mv2[['title', 'weight_average', 'vote_average']].sort_values(by='weight_average', ascending=False))







# %%
