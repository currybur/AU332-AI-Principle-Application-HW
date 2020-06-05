import numpy as np
import pandas as pd
from functools import reduce
from decimal import *

def findVarDis(bayesNet, var):
    df = pd.DataFrame()
    df['probs'] = [1]
    for table in bayesNet:
        if list(table.keys())[-1] == var:
            return table
    return df

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        # print(col)
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs
    # print(factorTable)
    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## factor1, factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for merging two factors
def joinFactors(factor1, factor2):
    # your code
    res = []

    names1 = set(factor1.keys()[1:])
    names2 = set(factor2.keys()[1:])
    shared = names1.intersection(names2)

    if names1==names2:
        return factor1
    if len(shared) == 0:
        for i in range(factor1.shape[0]):
            for j in range(factor2.shape[0]):
                c = []
                c.append(float(Decimal(str(factor1.iloc[i].values[0]))*Decimal(str(factor2.iloc[j].values[0]))))
                for k in list(factor1.iloc[i][1:]):
                    c.append(int(k))
                for k in list(factor2.iloc[j][1:]):
                    c.append(int(k))
                res.append(c)
        # print(res)
        res = np.array(res)
        df = pd.DataFrame()
        df['probs'] = res[:,0]
        ct = 1
        for name in factor1.keys()[1:]:
            df[name] = [int(i) for i in res[:,ct]]
            ct += 1
        for name in factor2.keys()[1:]:
            df[name] = [int(i) for i in res[:,ct]]
            ct += 1
        # print(df)
        return df

    df = pd.merge(factor1, factor2, on=list(shared))
    probs = df['probs_x']*df['probs_y']
    df = df.drop(['probs_x', 'probs_y'], axis=1)
    df.insert(0, 'probs', probs)

    return df


## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    # your code
    newdf = pd.DataFrame()
    names = list(factorTable.keys())[1:]
    try:
        names.remove(hiddenVar)
    except:
        return factorTable
    if names == []:
        newdf['probs'] = [1]
        return newdf

    mag=factorTable.groupby(names).mean()
    newdf['probs'] =mag['probs'].values

    for i in range(len(names)):
        try:
            newdf[names[i]] = [mag.index.levels[i][j] for j in mag.index.codes[i]]
        except:
            newdf[names[i]] = mag.index
    return newdf

## Marginalize a list of variables
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVars):
    # your code
    res = bayesNet

    for var in hiddenVars:
        df = pd.DataFrame()
        df['probs'] = [1]
        new_res = []
        for factor in res:
            if var in factor.keys():
                df = joinFactors(df, factor)
            else:
                new_res.append(factor)
        df = marginalizeFactor(df, var)
        new_res.append(df)
        res = new_res
    return res

## Update BayesNet for a set of evidence variables
## bayesNet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):
    # your code
    res = []
    for i in range(len(bayesNet)):
        table = bayesNet[i]
        for j in range(len(evidenceVars)):
            if evidenceVars[j] in table.keys():
                table = table[table[evidenceVars[j]]==evidenceVals[j]]

        res.append(table)
    return res


## Run inference on a Bayesian network
## bayesNet: a list of factor tables and each table is in dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using
## join and marginalization of the sets of variables.
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    # your code


    newNet = evidenceUpdateNet(marginalizeNetworkVariables(bayesNet, hiddenVar), evidenceVars, evidenceVals)
    res = newNet[0]
    for i in range(1,len(newNet)):
        try:
            res = joinFactors(res, newNet[i])
        except:
            print("error", newNet)
    total = sum(list(res['probs']))
    newProb = []
    for i in range(res.shape[0]):
        newProb.append(list(res['probs'])[i]/total)
    res['probs'] = newProb
    res.index = [i for i in range(res.shape[0])]
    return res

