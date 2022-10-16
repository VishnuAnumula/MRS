from django.shortcuts import render
from django.http import HttpResponse
from .models import TwoD

def index(request):
    return render(request, 'index.html')

def add(request):
    n = str(request.POST['num1']) 
    res=[]

    #KNN
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import spatial
    import operator

    # Accessing the dataset
    temp='static/Movies.csv'

    # Parsing the csv (dataset)
    m=pd.read_csv(temp)

    # Cleaninng (formatting) the genres column
    m['Genre']=m['Genre'].str.split(', ')
    # Unique genres classification
    gl=[]
    for ind, r in m.iterrows():
        genre=r['Genre']
        for g in genre:
            if g not in gl:
                gl.append(g)
    print(gl)
    # Binary list, Ex: for [Action, Comdedy, Romance] and movie [Romance]
    # then binary list is [0, 0, 1]
    def glbinary(g_l):
        glb=[]
        for g in gl:
            if g in g_l:
                glb.append(1)
            else:
                glb.append(0)
        return glb 
    # apply()-allows applying df to a func, lambda-anonymous func.
    m['Genre_bin']=m['Genre'].apply(lambda x: glbinary(x))
    print(m['Genre_bin'].head())

    # For Cast column similar to above steps
    m['Cast']=m['Cast'].str.split('\n')
    cl=[]
    for ind, r in m.iterrows():
        cast=r['Cast']
        for c in cast:
            if c not in cl:
                cl.append(c)
    def clbinary(c_l):
        clb=[]
        for c in cl:
            if c in c_l:
                clb.append(1)
            else:
                clb.append(0)
        return clb 
    m['Cast_bin']=m['Cast'].apply(lambda x: clbinary(x))
    print(m['Cast_bin'].head())

    # Now same for directors
    m['Director']=m['Director'].str.split('\n')
    dl=[]
    for ind, r in m.iterrows():
        dir=r['Director']
        for d in dir:
            if d not in dl:
                dl.append(d)
    def dlbinary(d_l):
        dlb=[]
        for d in dl:
            if d in d_l:
                dlb.append(1)
            else:
                dlb.append(0)
        return dlb
    m['Dir_bin']=m['Director'].apply(lambda x: dlbinary(x))
    print(m['Dir_bin'].head())

    # Cosine similarity, a.b=|a||b|cosθ (scalar product)
    print(glbinary(m['Genre'][1]))
    print(clbinary(m['Cast'][5]))
    print(dlbinary(m['Director'][3]))

    def Similarity(mid1, mid2):
        a=m.iloc[mid1]
        b=m.iloc[mid2]

        gA = a['Genre_bin']
        gB = b['Genre_bin']
        genreDist = spatial.distance.cosine(gA, gB)

        scA=a['Cast_bin']
        scB=b['Cast_bin']
        scDist=spatial.distance.cosine(scA, scB)
        
        dA = a['Dir_bin']
        dB = b['Dir_bin']
        dirDist = spatial.distance.cosine(dA, dB)
        
        return genreDist+dirDist+scDist

    # Score Predictor
    def ScPred():
        new_movie=n
        def getNeighbors(baseMovie, K):
            distances=[]
            for ind, movie in m.iterrows():
                if movie['Name']!=baseMovie:
                    dist=Similarity(m[m['Name']==baseMovie].index[0], ind)
                    distances.append((ind, dist))
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
            for x in range(K):
                neighbors.append(distances[x][0])
            return neighbors
        
        K = 10
        avgRating = 0
        neighbors = getNeighbors(new_movie, K)
        for neighbor in neighbors:
            mc=m.iloc[neighbor]
            t=TwoD()
            t.name=str(mc['Name'])+" ("+str(mc['Year'])+")"
            t.link=str(mc['URL'])
            t.rating=str(mc['Rating'])
            res.append(t)

    ScPred()

    return render(request, "result.html", {'result': res})

def back(request):
    return render(request, 'index.html')