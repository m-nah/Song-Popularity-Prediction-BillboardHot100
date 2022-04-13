#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 01:59:30 2022

@author: Marium Hassan 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#### 1A. Importing Billboard Chart Data
billboard = pd.read_csv("/Users/owner/Desktop/DS8007 visualization project/kcmillersean-billboard-hot-100-1958-2017/Hot Stuff.csv")


"""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 327895 entries, 0 to 327894
Data columns (total 10 columns):
 #   Column                  Non-Null Count   Dtype  
---  ------                  --------------   -----  
 0   url                     327895 non-null  object 
 1   WeekID                  327895 non-null  object 
 2   Week Position           327895 non-null  int64  
 3   Song                    327895 non-null  object 
 4   Performer               327895 non-null  object 
 5   SongID                  327895 non-null  object 
 6   Instance                327895 non-null  int64  
 7   Previous Week Position  295941 non-null  float64
 8   Peak Position           327895 non-null  int64  
 9   Weeks on Chart          327895 non-null  int64  
dtypes: float64(1), int64(4), object(5)"""

### The Hot Stuff.csv file contains data from the Billboard Hot 100 chart, spanning 1958 - 2021. It contains the following values
# Billboard Chart URL 

# WeekID - which week the chart was releaseed; in DD/MM/YY format
# billboard["WeekID"].nunique()
# Out[40]: 3279 > the billboard chart is released weekly for 63 years ~ 3276 weeks and so the number of week IDs is what we would expect

plt.style.use('dark_background')
billboard["WeekID"] = billboard['WeekID'].astype('string')
billboard.WeekID = billboard.WeekID.str.replace(' ','')

billboard.WeekID = pd.to_datetime(billboard.WeekID, format= '%m/%d/%Y', errors='ignore')


#billboard.info()

#billboard["year"] = billboard['WeekID'].dt.year
#billboard['month'] = billboard['WeekID'].dt.month
#billboard['day'] = billboard['WeekID'].dt.day

#billboard['WeekID'] = billboard["WeekID"].dt.date

# Song name
# billboard["Song"].nunique()
# Out[42]: 24360 > there are only a total of 24360 songs that have charted on the Hot 100 compared to 327,900 total entries. This makes sense as popular music tends to stay on the charts long term, and though we see fluctuations in rank, the songs that stay on the chart are quite similar, with new entries coming and going.

# Performer name
# billboard["Performer"].nunique()
# Out[43]: 10061 > many popular artists tend to have 

# SongID - Concatenation of song & performer

# Current week on chart

# Instance

# this is used to separate breaks on the chart for a given song. Example, an instance of 6 tells you that this is the sixth time this song has appeared on the chart

# Previous week position

# Peak Position (as of the corresponding week)

# Weeks on Chart (as of the corresponding week)

### as we can see there are 327,895 non-null values across all values aside from Previous Week Position, which makes sense as songs entering the chart for the first time will not have a previous position assigned to them


#### Visualizing Billboard Data

### Who are the top 25 songs of all time in terms of total weeks spent on the chart?
top25_weeks = billboard[["Performer", "Song","Weeks on Chart"]]

top25_weeks = top25_weeks.groupby(["Performer", "Song"]).max().nlargest(25, "Weeks on Chart").reset_index()

plt.figure(figsize=(15,10))
ax = sns.barplot(y = "Song", x = "Weeks on Chart", data = top25_weeks, color = "white" )
plt.title("Top 25 Songs by Number of Weeks Spent on Billboard Hot 100")


"""
Performer                                 Song                                         Weeks on Chart      
Imagine Dragons                           Radioactive                                      87
AWOLNATION                                Sail                                             79
Jason Mraz                                I'm Yours                                        76
The Weeknd                                Blinding Lights                                  76
LeAnn Rimes                               How Do I Live                                    69
LMFAO Featuring Lauren Bennett & GoonRock Party Rock Anthem                                68
OneRepublic                               Counting Stars                                   68
Adele                                     Rolling In The Deep                              65
Jewel                                     Foolish Games/You Were Meant For Me              65
Carrie Underwood                          Before He Cheats                                 64
"""

### Which performers have the most combined weeks spent on the chart? This will include all songs by the artist even if they are charting concurrently.

top25_performers = billboard[["Performer", "Song","Weeks on Chart"]]
top25_performers = top25_performers.groupby(["Performer", "Song"]).max().reset_index()

top25_performers = top25_performers.groupby(["Performer"]).sum().nlargest(25, "Weeks on Chart").reset_index()

plt.figure(figsize=(15,10))
ax = sns.barplot(y = "Performer", x = "Weeks on Chart", data = top25_performers, color = "white" )
plt.title("Top 25 Performers by Number of Collective Weeks Spent on Billboard Hot 100")

"""             Performer  Weeks on Chart
0         Taylor Swift            1022
1           Elton John             889
2              Madonna             857
3        Kenny Chesney             758
4                Drake             746
5           Tim McGraw             731
6          Keith Urban             673
7        Stevie Wonder             659
8          Rod Stewart             657
9         Mariah Carey             621
10     Michael Jackson             611
11             Chicago             607
12       Rascal Flatts             604
13          Billy Joel             588
14         The Beatles             585
15  The Rolling Stones             585
16     Aretha Franklin             569
17             Rihanna             566
18     Whitney Houston             561
19                P!nk             560
20        Brad Paisley             559
21        Jason Aldean             559
22       George Strait             553
23        Neil Diamond             553
24    Carrie Underwood             541
"""

### What is the song that was at #1 longest? What are some songs that charted at #1 for a long time?
a = billboard
a = a.loc[a["Week Position"] == 1, ["Song", "Performer", "Week Position"]]
a = a.groupby(["Song", "Performer"]).sum().nlargest(25, "Week Position").reset_index()

"""            Song  Week Position
0  Old Town Road             19"""

"""
	Song	Performer	Week Position
0	Old Town Road	Lil Nas X Featuring Billy Ray Cyrus	19
1	Despacito	Luis Fonsi & Daddy Yankee Featuring Justin Bieber	16
2	One Sweet Day	Mariah Carey & Boyz II Men	16
3	Candle In The Wind 1997/Something About The Way You Look Tonight	Elton John	14
4	I Gotta Feeling	The Black Eyed Peas	14
5	I Will Always Love You	Whitney Houston	14
6	I'll Make Love To You	Boyz II Men	14
7	Macarena (Bayside Boys Mix)	Los Del Rio	14
8	Uptown Funk!	Mark Ronson Featuring Bruno Mars	14
9	We Belong Together	Mariah Carey	14
10	"End Of The Road (From ""Boomerang"")"	Boyz II Men	13
11	The Boy Is Mine	Brandy & Monica	13
12	Blurred Lines	Robin Thicke Featuring T.I. + Pharrell	12
13	Boom Boom Pow	The Black Eyed Peas	12
14	Closer	The Chainsmokers Featuring Halsey	12
15	Lose Yourself	Eminem	12
16	See You Again	Wiz Khalifa Featuring Charlie Puth	12
17	Shape Of You	Ed Sheeran	12
18	Smooth	Santana Featuring Rob Thomas	12
19	Yeah!	Usher Featuring Lil Jon & Ludacris	12
20	God's Plan	Drake	11
21	I Swear	All-4-One	11
22	I'll Be Missing You	Puff Daddy & Faith Evans Featuring 112	11
23	Independent Women Part I	Destiny's Child	11
24	The Box	Roddy Ricch	11
"""
### Chart the position of the longest charting song, Radioactive by Imagine Dragons
## fun aside, turns out there are multiple Billboard hits titled "Radioactive" by a lot of different artists hence the need to sort by song and performer
song_chart = billboard[(billboard.Song == "Radioactive") & (billboard.Performer == "Imagine Dragons")]


plt.figure(figsize=(30,10))

s0_ax = sns.lineplot(y = "Week Position", x = "WeekID", data = song_chart, hue = "Performer")
s0_ax.invert_yaxis()
plt.xlabel('Date')
plt.title("The longest charting song position over time")

song_chart1 = billboard[(billboard.Song == "Old Town Road") & (billboard.Performer == "Lil Nas X Featuring Billy Ray Cyrus")]


plt.figure(figsize=(30,10))
plt.style.use('dark_background')
s1_ax = sns.lineplot(y = "Week Position", x = "WeekID", data = song_chart1)
s1_ax.invert_yaxis()
plt.xlabel('Date')
plt.title("The longest charting #1 song position over time")

### The second csv file contains attributes from these songs from the Spotify database
attributes = pd.read_csv("/Users/owner/Desktop/DS8007 visualization project/kcmillersean-billboard-hot-100-1958-2017/Hot 100 Audio Features.csv")


"""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 29503 entries, 0 to 29502
Data columns (total 22 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   SongID                     29503 non-null  object 
 1   Performer                  29503 non-null  object 
 2   Song                       29503 non-null  object 
 3   spotify_genre              27903 non-null  object 
 4   spotify_track_id           24397 non-null  object 
 5   spotify_track_preview_url  14491 non-null  object 
 6   spotify_track_duration_ms  24397 non-null  float64
 7   spotify_track_explicit     24397 non-null  object 
 8   spotify_track_album        24391 non-null  object 
 9   danceability               24334 non-null  float64
 10  energy                     24334 non-null  float64
 11  key                        24334 non-null  float64
 12  loudness                   24334 non-null  float64
 13  mode                       24334 non-null  float64
 14  speechiness                24334 non-null  float64
 15  acousticness               24334 non-null  float64
 16  instrumentalness           24334 non-null  float64
 17  liveness                   24334 non-null  float64
 18  valence                    24334 non-null  float64
 19  tempo                      24334 non-null  float64
 20  time_signature             24334 non-null  float64
 21  spotify_track_popularity   24397 non-null  float64
dtypes: float64(14), object(8) """

# acousticness- A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
# analysis_url- A URL to access the full audio analysis of this track. An access token is required to access this data.
# danceability- Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
# duration_ms- The duration of the track in milliseconds.
# energy- Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# id- The Spotify ID for the track.
# instrumentalness- Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
# key- The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
# liveness- Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
# loudness- The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
# mode- Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
# speechiness- Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
# tempo- The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
# time_signature- An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4".
# track_href- A link to the Web API endpoint providing full details of the track.
# url- The Spotify URL for the track.
# valence- A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).


### for purposes of analysis we will remove instances of Nan as we need to analyse attributes to see what makes a song popular, if no attributes are measured, cannot really use songs
attributes = attributes.dropna()
"""attributes['time_signature'] = attributes['time_signature'].astype(int).astype(str)

attributes['mode'] = attributes['mode'].astype(int).astype(str)
attributes['key'] = attributes['key'].astype(int).astype(str)"""
#attributes.info()

b = attributes[["instrumentalness", "speechiness", "liveness", "acousticness","tempo" , "danceability", "valence","energy", "loudness"]]
x = b.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
c = pd.DataFrame(x_scaled)
c.columns = ["instrumentalness", "speechiness", "liveness", "acousticness","tempo" , "danceability", "valence","energy", "loudness"] 


plt.figure(figsize=(30,10))
a_plot = sns.boxplot(data = c, color = "white", showmeans=True)
a_plot.set_xticklabels(["instrumentalness", "speechiness", "liveness", "acousticness","tempo" , "danceability", "valence","energy", "loudness"])
plt.title("Confidence measure attributes of Billboard Hot 100 Songs")

### Are there any correlated variables?
#a_pair = sns.pairplot(c, kind="reg")
plt.figure(figsize=(30,10)) 
a_corr = sns.heatmap(c.corr(), cmap= "coolwarm")

### Merge the two databases together
billboard['WeekID'] = billboard["WeekID"].dt.date
billboard["WeekID"] = billboard['WeekID'].astype('string')
audio = pd.merge(billboard, attributes)
learn_audio = audio
# now we keep instances of only number one hits

audio = audio[audio["Week Position"]==1]

audio = audio.drop(columns=['url', 'SongID', 'Instance', 'spotify_genre', 'spotify_track_id', 'spotify_track_preview_url', 'spotify_track_explicit', 'spotify_track_album', 'spotify_track_popularity', 'Peak Position'])
plt.figure(figsize=(30,10))
num = audio[["instrumentalness", "speechiness", "liveness", "acousticness","tempo" , "danceability", "valence","energy", "loudness"]]

x = num.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
num = pd.DataFrame(x_scaled)
num.columns = ["instrumentalness", "speechiness", "liveness", "acousticness","tempo" , "danceability", "valence","energy", "loudness"]
n_1 = sns.boxplot(data = num, color = "white", showmeans=True)
plt.title("Confidence measure attributes of Number 1 Billboard Hot 100 Songs")


### unsupervised learning: Hierarchical clustering

### we have to encode str/object labels to normalize them and get rid of non essential labels
#learn_audio.info()
""" #   Column                     Non-Null Count   Dtype         
---  ------                     --------------   -----         
 0   url                        168358 non-null  object        
 1   WeekID                     168358 non-null  datetime64[ns]
 2   Week Position              168358 non-null  int64         
 3   Song                       168358 non-null  object        
 4   Performer                  168358 non-null  object        
 5   SongID                     168358 non-null  object        
 6   Instance                   168358 non-null  int64         
 7   Previous Week Position     152714 non-null  float64       
 8   Peak Position              168358 non-null  int64         
 9   Weeks on Chart             168358 non-null  int64         
 10  year                       168358 non-null  int64         
 11  spotify_genre              168358 non-null  object        
 12  spotify_track_id           168358 non-null  object        
 13  spotify_track_preview_url  168358 non-null  object        
 14  spotify_track_duration_ms  168358 non-null  float64       
 15  spotify_track_explicit     168358 non-null  object        
 16  spotify_track_album        168358 non-null  object        
 17  danceability               168358 non-null  float64       
 18  energy                     168358 non-null  float64       
 19  key                        168358 non-null  float64       
 20  loudness                   168358 non-null  float64       
 21  mode                       168358 non-null  float64       
 22  speechiness                168358 non-null  float64       
 23  acousticness               168358 non-null  float64       
 24  instrumentalness           168358 non-null  float64       
 25  liveness                   168358 non-null  float64       
 26  valence                    168358 non-null  float64       
 27  tempo                      168358 non-null  float64       
 28  time_signature             168358 non-null  float64       
 29  spotify_track_popularity   168358 non-null  float64"""
 
#learn_audio.columns
"""['url', 'WeekID', 'Week Position', 'Song', 'Performer', 'SongID',
       'Instance', 'Previous Week Position', 'Peak Position', 'Weeks on Chart',
       'year', 'spotify_genre', 'spotify_track_id',
       'spotify_track_preview_url', 'spotify_track_duration_ms',
       'spotify_track_explicit', 'spotify_track_album', 'danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
       'spotify_track_popularity']"""

## remove unnecessary columns: 'url', 'SongID','Instance','spotify_genre', 'spotify_track_id','spotify_track_preview_url','spotify_track_explicit', 'spotify_track_album','spotify_track_popularity'
learn_audio = learn_audio.drop(columns=['url', 'Song',
       'Instance', 'spotify_genre', 'spotify_track_id',
       'spotify_track_preview_url',
       'spotify_track_explicit', 'spotify_track_album',
       'spotify_track_popularity'])

#learn_audio.info()
"""   Column                     Non-Null Count   Dtype         
---  ------                     --------------   -----         
 0   WeekID                     168358 non-null  string
 1   Week Position              168358 non-null  int64         
 2   Performer                  168358 non-null  object        
 3   SongID                     168358 non-null  object        
 4   Previous Week Position     152714 non-null  float64       
 5   Peak Position              168358 non-null  int64         
 6   Weeks on Chart             168358 non-null  int64         
 7   year                       168358 non-null  int64         
 8   spotify_track_duration_ms  168358 non-null  float64       
 9   danceability               168358 non-null  float64       
 10  energy                     168358 non-null  float64       
 11  key                        168358 non-null  float64       
 12  loudness                   168358 non-null  float64       
 13  mode                       168358 non-null  float64       
 14  speechiness                168358 non-null  float64       
 15  acousticness               168358 non-null  float64       
 16  instrumentalness           168358 non-null  float64       
 17  liveness                   168358 non-null  float64       
 18  valence                    168358 non-null  float64       
 19  tempo                      168358 non-null  float64       
 20  time_signature             168358 non-null  float64    """

#learn_audio.columns
"""Index(['WeekID', 'Week Position', 'Performer', 'SongID',
       'Previous Week Position', 'Peak Position', 'Weeks on Chart',
       'spotify_track_duration_ms', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'time_signature'],
      dtype='object')

"""
### define popularity as a song that has has a peak position of less than 50, and been on the chart longer than 10 weeks


learn_audio = learn_audio.loc[(learn_audio["Peak Position"] < 50) & (learn_audio["Weeks on Chart"] > 20)]

learn_audio["Peak Position"] = (learn_audio["Peak Position"] == 1).astype(int)


learn_audio = learn_audio.groupby(['SongID','Performer', 'Peak Position',
       'spotify_track_duration_ms', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'time_signature'])["Weeks on Chart"].max().reset_index()

### labels
y = learn_audio["Peak Position"]

# encode non-numeric variables
le = preprocessing.LabelEncoder()
le.fit(learn_audio["SongID"])
learn_audio["SongID"] = le.transform(learn_audio["SongID"])

le.fit(learn_audio["Performer"])
learn_audio["Performer"] = le.transform(learn_audio["Performer"])


# normalize dataset 
learn_audio = preprocessing.scale(learn_audio)
learn_audio = pd.DataFrame(learn_audio)
learn_audio.columns = ['SongID', 'Performer', 'Peak Position',
       'spotify_track_duration_ms', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'time_signature', 'Weeks on Chart']

learn_audio = learn_audio.dropna()




X = learn_audio.drop(columns= ['Peak Position', "SongID", "Weeks on Chart", "Performer"])

### Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


### build classifiers

# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)

sgd.score(X_train, y_train)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

# random forest 
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)


# Results
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_log, 
              acc_random_forest, acc_gaussian, 
              acc_sgd, acc_decision_tree]})
results = results.sort_values(by='Score', ascending=False)
#result_df = result_df.set_index('Score')

plt.figure(figsize=(30,10))
res_plot = sns.barplot(x = 'Model', y = 'Score', data = results, color = 'white')
plt.title("Accuracy of Predictive Models")

plt.figure(figsize=(30,10))
res_plot = sns.barplot(x = 'feature', y = 'importance', data = importances, color = 'white')
plt.title("Most important features in predicting if a song would become a Number One Hit")