import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType, IntegerType
from pyspark.mllib.linalg.distributed import RowMatrix
import csv

#-----------------------------------------------------------MODIFICATIONS POUR LE FICHIER GENRE
#-----------------------------------------------------------CREATION D'UN NOUVEAU FICHIER

nameByGenre = dict()

with open("name_gender_dataset.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        #si le nom existe déjà : I : indéterminé

        if(nameByGenre.get(row[0])!=None):
            nameByGenre[row[0]] = "I"
        else:
            nameByGenre[row[0]]=row[1]

#creation new fichier
with open("nameByGenre.csv","w",newline="",encoding="utf-8") as csvfile:
    fieldnames = ['name','genre']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for cle, valeur in nameByGenre.items():
        writer.writerow({'name':cle, 'genre':valeur})

#-----------------------------------------------------------PARTIE SPARK
spark = SparkSession\
    .builder\
    .appName("genderName")\
    .getOrCreate()

sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)

schema = StructType([
    StructField("Name", StringType()),
    StructField("Gender", StringType()),
    StructField("Count", IntegerType()),
    StructField("Probability", FloatType()),
    StructField("genreFinal", StringType())
])

genres = sqlContext.read.csv("name_gender_dataset.csv", header=False, schema=schema)



