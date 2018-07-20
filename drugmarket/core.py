# -*- coding: utf-8 -*-

from pymongo import MongoClient
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tabulate import tabulate
from helpers import getlisted, mgtagger, phasecounts, marketcap
getlisted()
mgtagger()
phasecounts()
marketcap()

def runCore():

    client = MongoClient("mongodb://localhost:27017")
    db_stocks = client.stocks

    cursor = db_stocks.listed.find({
        "$and": [
            {"phaseCounts": {"$exists": True}},
            {"marketcap": {"$exists": True}}
        ]
    })

    comp = []
    sym = []
    p1s = []
    p2s = []
    p3s = []
    p4s = []
    mcap = []

    for li in cursor:
        comp.append(li['_name'][0:20])
        sym.append(li['_symbol'])
        p1s.append(li['phaseCounts']['Phase 1'])
        p2s.append(li['phaseCounts']['Phase 2'])
        p3s.append(li['phaseCounts']['Phase 3'])
        p4s.append(li['phaseCounts']['Phase 4'])
        mcap.append(li['marketcap'])

    data = {
        'Company': comp,
        'Symbol': sym,
        'Phase 1': p1s,
        'Phase 2': p2s,
        'Phase 3': p3s,
        'Phase 4': p4s,
        'MC': mcap
    }
    df1 = pd.DataFrame(data, columns=data.keys())
    df1 = df1.sort_values(['MC', 'Phase 3'], ascending=[True, False])
    # find stocks only with marketcap less than certain amount
    df2 = df1.loc[df1['MC'] < 700000000]

    print(tabulate(df2, headers='keys', tablefmt='psql'))

    # return dataframe to be used
    df1.to_csv(path_or_buf="drugmarket_dataframe.tsv", sep='\t', encoding='utf-8')
    return df1

    # Future data to get
    # - Prevalence of conditions in the US
    # - Current revenue of stocks
    # - Stocks broken up by their involvement with conditions
    # - Further broken down by phase and number of clinical trials for that condition
    # - Average price of drugs once approved per condition (use precedence like hep. C drug)

runCore()

# Analysis ideas

# Multi-variate linear regression to predict the marketcap based on the number of trials in pipeline..That way,
# if you can guess the future number of trials, then you can guess the future marketcap / stock price.
# This will not be very valuable, but do it just to practice linear regression.

# Then later, deep learning on many more factors including condition types, trial locations, etc.
