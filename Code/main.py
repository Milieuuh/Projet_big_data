import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
import csv

#-----------------------------------------------------------MODIFICATIONS POUR LE FICHIER GENRE
#-----------------------------------------------------------CREATION D'UN NOUVEAU FICHIER

nameByGenre = dict()
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            'à','ö','é','è','ü','ï','î']

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

#creation de tableaux avec les différentes terminaison supposées
terminaisonMasculin = ["o", "i", "k", "d", "s", "r", "t", "n", "m", "b", "z", "l", "w"]
terminaisonFeminin = ["a", "e", "h"]


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
    inputCols=['len', 'terminaison'],
    outputCol="features")
genres_with_features = vecAssembler.transform(genres)



# Do K-means
k = 2
kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(genres_with_features)
centers = model.clusterCenters()

genres_with_clusters = model.transform(genres_with_features)
print("Centers", centers)

# Convert Spark Data Frame to Pandas Data Frame
genres_for_viz = genres_with_clusters.toPandas()

# Vizualize
genreF = genres_for_viz['genre'] == 'F'
genreFem = genres_for_viz[genreF]
genreM = genres_for_viz['genre'] == 'M'
genreMas = genres_for_viz[genreM]

# Colors code k-means results, cluster numbers
colors = {0: 'red', 1: 'green'}

fig = plt.figure().gca(projection="rectilinear")

# triangle pour les gars
fig.scatter(genreMas.len,
            genreMas.terminaison,
            c=genreMas.prediction.map(colors),
            marker='v')

# carré pour les filles
fig.scatter(genreFem.len,
            genreFem.terminaison,
             c=genreFem.prediction.map(colors),
             marker='s')
fig.set_xlabel('Longueur du prénom')
fig.set_ylabel('terminaison du prénom')
plt.show()
