#!/usr/bin/env python
# coding: utf-8

# # A content based movie recommender system using cosine similarity

# # Import Libraries

# In[56]:


import pandas as pd
import numpy as np
import ast


# Import the Dataset

# In[57]:


movies = pd.read_csv('imdb_data - imdb_data.csv')


# In[58]:


movies.head()


# In[59]:


movies.head(1)['cast'].values


# In[60]:


movies.head(1)['Keywords'].values


# In[61]:


movies.shape


# In[62]:


movies['original_language'].value_counts()


# In[63]:


movies.columns


# In[64]:


movies = movies[['id', 'imdb_id', 'title', 'overview', 'genres', 'Keywords', 'cast', 'crew']]


# In[65]:


movies.head()


# In[66]:


movies.isnull().sum()


# In[67]:


movies.duplicated().sum()


# In[68]:


movies.head(1)


# In[13]:


type(movies['genres'][0])


# In[14]:


movies.iloc[3].genres


# In[16]:


movies.iloc[0].overview


# In[18]:


movies.iloc[0].Keywords


# In[1]:


import ast
# ast.literal_eval("[{'id': 53, 'name': 'Thriller'}, {'id': 18, 'name': 'Drama'}]")


# Handle Missing Values Of Genres

# In[69]:


def convert(obj):
    if pd.isna(obj):
        return []
    else:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L


# In[70]:


movies['genres'] = movies['genres'].apply(convert)


# In[71]:


type(movies['genres'][0])


# In[72]:


movies.head()


# In[79]:


movies['Keywords'] = movies['Keywords'].apply(convert)


# In[80]:


movies.head()


# In[73]:


def convert2(obj):
    if pd.isna(obj):
        return []
    else:
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter+=1
            else:
                break
        return L


# In[74]:


movies['cast'] = movies['cast'].apply(convert2)


# In[75]:


movies.head()


# In[52]:


movies.iloc[0].crew


# In[ ]:


def fetch_director(obj):
    if pd.isna(obj):
        return []
    else:
        L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])    
            break
    return L        
    


# In[77]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[81]:


movies.head()


# In[51]:


movies['overview'][0]


# In[37]:


type(movies['overview'][0])


# In[3]:


movies['overview'] = movies['overview'].apply(lambda x: x.split() if pd.notna(x) and np.isscalar(x) else [])


# In[84]:


movies.head()


# In[85]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "")for i in x])
movies['Keywords'] = movies['Keywords'].apply(lambda x: [i.replace(" ", "")for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "")for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "")for i in x])


# In[86]:


movies.head()


# In[90]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['Keywords'] + movies['cast'] + movies['crew']


# In[91]:


movies.head()


# In[92]:


movies['tags'][0]


# In[88]:


movies.isnull().sum()


# In[96]:


new_df = movies[['imdb_id', 'title', 'tags']]


# In[97]:


new_df


# In[98]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[99]:


new_df.head()


# In[100]:


new_df['tags'][0]


# In[101]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[102]:


new_df.head()


# In[51]:


pip install nltk


# In[52]:


import nltk


# In[103]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[104]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[115]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[116]:


new_df['tags'][1] 


# In[117]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000, stop_words = 'english')


# In[118]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[119]:


vectors


# In[120]:


vectors[0]


# In[121]:


cv.get_feature_names()


# In[122]:


len(cv.get_feature_names())


# In[123]:


from sklearn.metrics.pairwise import cosine_similarity


# In[125]:


cosine_similarity(vectors).shape


# In[126]:


cosine_similarity(vectors)


# In[127]:


similarity = cosine_similarity(vectors)


# In[128]:


similarity[1].shape


# In[129]:


similarity[0]


# In[132]:


new_df[new_df['title'] == 'Kahaani'].index[0]


# In[133]:


sorted(list(enumerate(similarity[0])),reverse = True, key= lambda x:x[1])[1:11]


# In[134]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True,key= lambda x:x[1])[1:11]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[137]:


recommend('Kahaani')


# In[136]:


new_df.iloc[90].title


# In[74]:


import pickle


# In[75]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[76]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




