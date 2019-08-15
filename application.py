from flask import Flask, redirect, render_template, request, make_response
import pymysql.cursors
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from collections import Counter, defaultdict
import math
import itertools
import time
import random

application = Flask(__name__)

endpoint = "bhanusqldb.cpublxbeijbp.us-east-2.rds.amazonaws.com"
port = "3306"
user = "bhanusqldb"
password = "bhanusqldb"
database = "bhanusqldb"


def connectDb():
    connection = pymysql.connect(host=endpoint,
                                 # port=port,
                                 user=user,
                                 password=password,
                                 db=database,
                                 local_infile=True
                                 )
    cursor = connection.cursor()
    return cursor


@application.route('/')
def index():
    return render_template('index.html')

@application.route('/searchByYear')
def searchByYear():
    cursor = connectDb()
    year = request.args.get('year')
    population = (request.args.get('population'))

    sql = "SELECT State, `" + year + "` FROM population where `" + year + "` >" + population

    print(sql)
    starttime = time.time()
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)

    end_time = time.time()
    duration = end_time - starttime

    return render_template('population.html', ci=result, time=duration)


@application.route('/searchMultiple')
def searchMultiple():
    cursor = connectDb()
    year = request.args.get('year')
    count = int(request.args.get('count'))
    # population = request.args.get('population')


    starttime = time.time()
    for i in range(0,int(count)):
          random1 = random.randint(1000000, 30000000)
          print(random1)
          sql = "select State,'" + str(year) + "' from population where '" + str(year) + "' > '" + str(random1) + "'"

          print(sql)
          cursor.execute(sql)

    rows=[]
    rows = cursor.fetchall()
    print(rows)
    endtime = time.time()
    duration = endtime - starttime
    return render_template('population.html', ci=rows, time=duration)


@application.route('/kmeansClusteringCSV')
def kmeansClusteringCSV():
    numberOfClusters = int(request.args.get('numberOfClusters'))
    column1 = request.args.get('column1')
    column2 = request.args.get('column2')

    data_frame = pd.read_csv("titanic.csv")
    img = BytesIO()
    data_frame.head()
    # data_frame[['Age', 'Fare']].hist()
    # plt.show()
    x = data_frame[[column1, column2]].fillna(0)
    x = np.array(x)
    print(x)

    kmeans = KMeans(n_clusters=numberOfClusters)

    kmeansoutput_x = kmeans.fit(x)
    centroid = kmeans.cluster_centers_
    print(type(kmeansoutput_x))

    pl.figure('{} Cluster K-Means'.format(numberOfClusters))

    pl.scatter(x[:, 0], x[:, 1], cmap='rainbow')
    pl.scatter(centroid[:,0], centroid[:,1], marker="x", s=150, linewidths=5, zorder=10)

    pl.title('5 Cluster K-Means')
    pl.xlabel(column1)

    pl.ylabel(column2)

    # pl.show()

    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())

    response = make_response(img.getvalue())
    response.headers['Content-Type'] = 'image/png'

    return response


@application.route('/kmeansClusteringDisplay', methods=["GET","POST"])
def kmeansClusteringDisplay():

    numberOfClusters = int(request.form['numberOfClusters'])
    column1 = request.form['column1']
    column2 = request.form['column2']

    cursor = connectDb()
    sql = "select "+column1+", "+column2+" from AWS.titanic where(AWS.titanic.age is NOT NULL AND AWS.titanic.FARE is NOT NULL)"
    print(sql)
    starttime = time.time()

    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    print(result)

    data_frame = result
    img = BytesIO()
    # data_frame.head()
    # data_frame[['Age', 'Fare']].hist()
    # plt.show()
    # x = data_frame[[column1, column2]].fillna(0)
    x = np.array(result)

    print(x)
    #
    kmeans = KMeans(n_clusters=numberOfClusters)
    #
    kmeansoutput_x = kmeans.fit(x)
    centroid = kmeans.cluster_centers_
    print(type(kmeansoutput_x))
    #
    count = Counter(kmeans.labels_)
    print("I'm cluster counter", count)
    clusters_indices = defaultdict(list)
    for index, c in enumerate(kmeans.labels_):
        clusters_indices[c].append(x[index])

    print(clusters_indices)
    print(clusters_indices[1][0][0])
    print(clusters_indices[1][0][1])
    count = Counter(kmeans.labels_)

    end_time = time.time()
    duration = end_time-starttime

    pl.figure('K-Means')
    pl.scatter(x[:, 0], x[:, 1], cmap='rainbow')
    pl.scatter(centroid[:, 0], centroid[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    pl.title(str(numberOfClusters) + str(Counter(kmeans.labels_)))
    pl.xlabel(column1)
    pl.ylabel(column2)

    img = BytesIO()
    plt.savefig(img, format='png')
    fig2 = plt.figure()
    axes = fig2.add_subplot(111)  # Add subplot (dont worry only one plot appears)
    axes.set_autoscale_on(True)  # enable autoscale
    axes.autoscale_view(True, True, True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue())
    response = make_response(img.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response




@application.route('/kmeansClustering', methods=["GET","POST"])
def kmeansClustering():

    numberOfClusters = int(request.form['numberOfClusters'])
    column1 = request.form['column1']
    column2 = request.form['column2']

    cursor = connectDb()
    sql = "select "+column1+", "+column2+" from AWS.titanic where(AWS.titanic.age is NOT NULL AND AWS.titanic.FARE is NOT NULL)"
    print(sql)
    starttime = time.time()

    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    print(result)

    data_frame = result
    img = BytesIO()
    # data_frame.head()
    # data_frame[['Age', 'Fare']].hist()
    # plt.show()
    # x = data_frame[[column1, column2]].fillna(0)
    x = np.array(result)

    print(x)
    #
    kmeans = KMeans(n_clusters=numberOfClusters)
    #
    kmeansoutput_x = kmeans.fit(x)
    centroid = kmeans.cluster_centers_
    print(type(kmeansoutput_x))
    #
    count = Counter(kmeans.labels_)
    print("I'm cluster counter", count)
    clusters_indices = defaultdict(list)
    for index, c in enumerate(kmeans.labels_):
        clusters_indices[c].append(x[index])

    print(clusters_indices)
    print(clusters_indices[1][0][0])
    print(clusters_indices[1][0][1])
    count = Counter(kmeans.labels_)

    end_time = time.time()
    duration = end_time-starttime
    return render_template('clusters.html',centroid=centroid, ci=clusters_indices, totalPoints=count, time=duration)



@application.route('/DistanceCentroidPoints', methods=["GET","POST"])
def DistanceCentroidPoints():

    numberOfClusters = int(request.form['numberOfClusters'])
    column1 = request.form['column1']
    column2 = request.form['column2']

    cursor = connectDb()
    sql = "select "+column1+", "+column2+" from AWS.titanic where(AWS.titanic.age is NOT NULL AND AWS.titanic.FARE is NOT NULL)"
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()

    data_frame = result
    img = BytesIO()
    x = np.array(result)
    # print(x)

    kmeans = KMeans(n_clusters=numberOfClusters)

    kmeansoutput_x = kmeans.fit(x)
    centroid = kmeans.cluster_centers_
    # print("centroids: ", centroid)
    # print(type(kmeansoutput_x))

    count = Counter(kmeans.labels_)
    print("I'm cluster counter", count)

    dataPoints = []

    clusters_indices = defaultdict(list)
    for index, c in enumerate(kmeans.labels_):
        clusters_indices[c].append(x[index])
        dataPoints.append(x[index])
        print("im point :",x[index])

    print(clusters_indices)
    # for calculating distance
    dist_list = []
    cordinate1 = []
    cordinate2 = []
    for i in range(0, len(centroid) - 1):
        for j in range(i + 1, len(centroid)):
            x1 = centroid[i][0]
            x2 = centroid[j][0]
            y1 = centroid[i][1]
            y2 = centroid[j][1]
            temp = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            dist = math.sqrt(temp)

            dist_list.append(list(zip(centroid[i][:], centroid[j][:], itertools.repeat(dist))))

    print(dist_list)
    dist_len = len(dist_list)

    return render_template('newclusters.html', ci=dist_list, totalPoints = dist_len)



@application.route('/persondetailsincluster', methods=["GET","POST"])
def persondetailsincluster():

    numberOfClusters = int(request.form['numberOfClusters'])
    column1 = request.form['column1']
    column2 = request.form['column2']
    clusternumber = request.form['numberOfClusters']
    cursor = connectDb()
    sql = "select "+column1+", "+column2+" from AWS.titanic where(AWS.titanic.age is NOT NULL AND AWS.titanic.FARE is NOT NULL)"
    print(sql)
    starttime = time.time()

    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)
    data_frame = result
    x = np.array(result)

    print(x)
    #
    kmeans = KMeans(n_clusters=numberOfClusters)
    #
    kmeansoutput_x = kmeans.fit(x)
    centroid = kmeans.cluster_centers_
    print(type(kmeansoutput_x))
    #
    count = Counter(kmeans.labels_)
    print("I'm cluster counter", count)
    clusters_indices = defaultdict(list)
    for index, c in enumerate(kmeans.labels_):
        clusters_indices[c].append(x[index])

    print(clusters_indices)
    print(clusters_indices[0][0])

    value1 =int(clusters_indices[1][0][0])
    value2 = clusters_indices[1][0][1]
    print(clusters_indices[1][0][1])
    query ="select AWS.titanic.age , AWS.titanic.cabin , AWS.titanic.survived from AWS.titanic where AWS.titanic.{} = {} and AWS.titanic.{} = {}".format(column1,value1,column2,value2)
    print(query)
    cursor.execute(query)
    result1 = cursor.fetchall()
    print(result1)
    # print(result1[0][0])
    cursor.close()
    connectDb()
    return render_template("persondetails.html" ,age =result1[0][0] ,cabin = result1[0][1] ,survived = result1[0][0] )


def calculatedistance(listofcordinates):
    distance = math.sqrt(( listofcordinates[0][0]  - listofcordinates[1][0])*2 + (listofcordinates[0][1]  - listofcordinates[1][1])*2 )
    return distance


if __name__ == '__main__':
    application.run()
