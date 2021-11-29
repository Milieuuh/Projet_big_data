import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
import numpy as np
from pyspark.ml import Pipeline
import csv
from pyspark.ml.evaluation import ClusteringEvaluator

#-----------------------------------------------------------MODIFICATIONS POUR LE FICHIER GENRE
#-----------------------------------------------------------CREATION D'UN NOUVEAU FICHIER

nameByGenre = dict()
alphabet = ['a', 'e', 'i', 'y', 'h', 'j', 'o', 'i', 'œ', 'c', 'f', 'g', 'h', 'j', 'p', 'q', 'u', 'v', 'w', 'x',
            'k', 'd', 's', 'r', 't', 'n', 'm', 'b', 'z', 'l', 'w','ö', 'à']

with open("name_gender_dataset.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        #si le nom existe déjà : I : indéterminé
        if(nameByGenre.get(row[0])!=None):
            #nameByGenre[row[0]] = "I"
            continue
        else:
            nameByGenre[row[0]]=row[1]

#creation new fichier
with open("nameByGenre.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ['#len', 'name', 'genre', 'terminaison']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cle, valeur in nameByGenre.items():
        char = cle[len(cle)-1].lower()
        if char.isalpha():
            writer.writerow({'#len': len(cle), 'name':cle, 'genre':valeur, "terminaison":alphabet.index(char)})

#-----------------------------------------------------------PARTIE SPARK

spark = SparkSession\
    .builder\
    .appName("genderName")\
    .getOrCreate()

sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

schema = StructType([
    StructField("len", IntegerType(), nullable=True), #Longueur du prénom
    StructField("name", StringType(), nullable=True),
    StructField("genre", StringType()),
    StructField("terminaison", IntegerType(), nullable=True)
])

genres = sqlContext.read.csv("nameByGenre.csv", header=False, schema=schema, comment='#')

genres.show()

vecAssembler = VectorAssembler(
    inputCols=['terminaison'],
    outputCol="features")
genres_with_features = vecAssembler.transform(genres)



# Do K-means
k = 2
kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(genres_with_features)
centers = model.clusterCenters()


genres_with_clusters = model.transform(genres_with_features)
print("Centers", centers)


evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(genres_with_clusters)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# Convert Spark Data Frame to Pandas Data Frame
genres_for_viz = genres_with_clusters.toPandas()

# Vizualize
genreF = genres_for_viz['genre'] == 'F'
genreFem = genres_for_viz[genreF]
genreM = genres_for_viz['genre'] == 'M'
genreMas = genres_for_viz[genreM]

# Colors code k-means results, cluster numbers
colors = {0: 'red', 1: 'green'}

def transforme(s):
    if s == 'M': return 0
    else: return 1

Genres = [ transforme(x) for x in genres_for_viz.genre ]


fig = plt.figure().gca(projection="rectilinear")

# triangle pour les gars
fig.scatter(genreMas.terminaison,
            genreMas.terminaison*0.2,
            c=genreMas.prediction.map(colors),
            marker='v')

# carré pour les filles
fig.scatter(genreFem.terminaison,
            [4] * len(genreFem.terminaison),
            c=genreFem.prediction.map(colors),
            marker='s')
fig.set_xlabel('Longueur du prénom')
fig.set_ylabel('terminaison du prénom')
plt.show()

fig = plt.figure().gca(projection="rectilinear")
fig.hist(genreMas.terminaison)
fig.set_title("Terminaison pour les hommes")
fig.set_xlabel('Terminaison')
fig.set_ylabel("nb d'hommes")

fig = plt.figure().gca(projection="rectilinear")
fig.hist(genreFem.terminaison)
fig.set_title("Terminaison pour les femmes")
fig.set_xlabel('Terminaison')
fig.set_ylabel("nb de femmes")

Pred = genres_for_viz.prediction
fig = plt.figure().gca(projection="rectilinear")

fig.hist2d(Genres,
             Pred, bins = 2
            )

#fig.hist2d(genres_for_viz.len, genres_for_viz.terminaison)

