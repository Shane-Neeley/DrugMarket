import pandas as pd
import numpy as np
import json
import io
import requests
from ftplib import FTP
import time
from decimal import Decimal
from pymongo import MongoClient
from datetime import date
import os
import subprocess

# set a global today, y-m-d so can sort
today = date.today().strftime('%Y-%m-%d')

########################################################

def getlisted():
    print('running getlisted')
    # '''read in the listed files into db'''
    client = MongoClient("mongodb://localhost:27017")
    # create database stocks
    db_stocks = client.stocks

    # get collection, delete it
    listed = db_stocks['listed']
    listed.remove({})

    ftp = FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("/SymbolDirectory/")

    def grabFile(fname):
        localfile = open(fname, 'wb')
        ftp.retrbinary('RETR ' + fname, localfile.write, 1024)
        localfile.close()

    grabFile("nasdaqlisted.txt")
    grabFile("otherlisted.txt")
    ftp.quit()

    df1 = pd.read_csv("nasdaqlisted.txt", sep='|')
    df2 = pd.read_csv("otherlisted.txt", sep='|')

    # make dictionarys out of dataframe for inserting to mongo
    r1 = json.loads(df1.T.to_json()).values()
    r2 = json.loads(df2.T.to_json()).values()

    records = []
    for i in r1:
        records.append(i)
    for i in r2:
        records.append(i)

    for i in records:

        # normalize - create internal symbol
        if 'Symbol' in i:
            i['_symbol'] = i['Symbol']
        elif 'NASDAQ Symbol' in i:
            i['_symbol'] = i['NASDAQ Symbol']

        # normalize - create internal name
        i['Security Name'] = i['Security Name'] or ""
        nogood = [
            "Limited American Depositary Shares each representing one hundred Ordinary Shares",
            " - Warrant",
            ' - ',
            "Common Stock", "(Antigua/Barbudo)", "(Canada)", "Common Shares",
            "Holding Corporation", "Holding Company", "Holding Corp", "(Holding Company)"
            "Incorporated", " Inc", "Class A", " Corporation", " Corp",
            " (Delaware)", " (DE)", " plc", " N.V.", " S.A.", " A/S", " NV",
            " SA", "Ordinary Shares", "Depositary Shares", " Ltd", " Limited", " AG"
            ",",
            ".",
            "()"
        ]
        i['_name'] = i['Security Name']
        for _ng in nogood:
            i['_name'] = i['_name'].replace(_ng, '')
        i['_name'] = i['_name'].strip()
        i['lastupdated'] = today

    listed.insert(records)

    print('ran listed')


########################################################

# Get latest data
def mmdata():
    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    bashCommand = "sh " + desktop + "/dump-restore.sh"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

########################################################

def mgtagger():
    # use medicalgroups name and synonyms to tag the stock listings
    print('running mg tagging ...')

    client = MongoClient("mongodb://localhost:27017")
    # to get this data, must buy license from http://api.molecularmatch.com
    db_stocks = client.stocks
    mgcursor = db_stocks.medicalgroup.find({'exclude': False})
    listed = db_stocks.listed

    # for each medical group in molecularmatch
    for mg in mgcursor:
        # Gather all it's names
        mg_names = [mg['name']]
        # Warning: synonyms can be very loose
        for syn in mg['synonyms']:
            if syn['suppress'] == False:
                mg_names.append(syn['name'])

        # For each name for this medicalgroup, look for a match within the formal security name
        for mgsyn in mg_names:
            re = "^" + mgsyn
            matches = list(listed.find({"Security Name": {'$regex': re}}))
            if len(matches) > 0:
                for m in matches:
                    # Save this to the listed collection
                    listed.update(
                        {'_id': m['_id']},
                        {'$addToSet': {'medicalgroups': mg['name']}}
                    )

    print('ran mgtagger')

########################################################

def marketcap():
    # Get a MC / trialcount number
    # This downloads the current market cap of the stock
    # https://www.quantshare.com/sa-426-6-ways-to-download-free-intraday-and-tick-data-for-the-us-stock-market
    print('running marketcap')
    db = MongoClient("mongodb://localhost:27017").stocks

    cursor = db.listed.find({"medicalgroups.0": {"$exists": True}})
    # for each stock with tagged medicalgroups
    for li in cursor:
        # download the market cap and save to the stock
        url = "https://api.iextrading.com/1.0/stock/" + li['_symbol'] + "/quote"
        with requests.Session() as s:
            download = s.get(url)
            content = json.loads(download.content.decode('utf-8'))
            db.listed.update(
                {'_id': li['_id']},
                {'$set': {
                    "marketcap": content['marketCap']
                    }
                }
            )

        # Annual financials .. operatingIncome = totalRevenue - operatingExpense
        # For companies with big ol revenue, subract 5x revenue from marketcap to get pipeline value?
        # https://api.iextrading.com/1.0/stock/NOVN/financials?period=annual
        valuationMultiplier = 5
        url2 = "https://api.iextrading.com/1.0/stock/" + li['_symbol'] + "/financials?period=annual"
        with requests.Session() as s:
            download2 = s.get(url2)
            content2 = json.loads(download2.content.decode('utf-8'))
            if "financials" in content2:
                lastupdatedfinacials = content2["financials"][0] #last reported date
                income = lastupdatedfinacials['operatingIncome']
                db.listed.update(
                    {'_id': li['_id']},
                    {'$set': {
                        "operatingincome": income
                        }
                    }
                )

                if income and income > 0:
                    adjustedMarketCap = content["marketCap"] - (valuationMultiplier * income)
                    db.listed.update(
                        {'_id': li['_id']},
                        {'$set': {"pipelineAdjustedMarketCap": adjustedMarketCap}}
                    )


    print('ran marketcap')


########################################################

def tagcounts():
    print('running tagcounts')

    client = MongoClient("mongodb://localhost:27017")
    # to get this data, must buy license from http://api.molecularmatch.com
    # get a unique list of medicalgroups to query trials for
    db_stocks = client.stocks
    listed = db_stocks['listed']

    # list the ones with most valuable trials, there may be outliers here like ,
    # healthcare companies with big cap and one trial ... i could find these like that too. big cap but few trials.
    avoid = [
        "National Institutes of Health",
        "National Cancer Institute",
        "Duke University",
        "Harvard University",
        "McKesson",
        "Quest Diagnostics Incorporated",
        "Thermo Fisher",
        "Stryker",
        "Becton, Dickinson and Company",
        "Intuitive Surgical",
        "Illumina, Inc.",
        "Agilent Technologies, Inc.",
        "3M Company",
        "ABIOMED, Inc.",
        "Edwards Life Sciences",
        "Smith & Nephew",
        "Haemonetics Corporation",
        "Masimo Corporation",
        "Catalent",
        "Cardinal Health",
        "Fresenius Medical Care",
        "Perrigo Company",
        "Laboratory Corporation of America",
        "ResMed, Inc.",
        "Medtronic"
    ]

    licursor = listed.find({
        "medicalgroups.0":{"$exists": True},
        "medicalgroups":{"$nin": avoid},
        "marketcap":{"$exists": True},
        "marketcap":{"$gt": 0}
    })

    mgs = []
    for li in licursor:
        for mg in li['medicalgroups']:
            mgs.append(mg)
    mgs = np.unique(mgs).tolist()

    # Any trials with a medicalgroup tag we want that are open
    q = {"$and": [
            {"tags.term": {"$in":mgs}},
            {"tags.facet": "PHASE"},
            {"tags.term": {"$ne":"Temporarily not available"}},
            {"tags.term": {"$ne":"Suspended"}},
            {"tags.term": {"$ne":"Closed"}},
            {"tags.term": {"$ne":"Completed"}},
            {"tags.term": {"$ne":"Withdrawn"}},
            {"tags.term": {"$ne":"Withheld"}},
            {"tags.term": {"$ne":"Terminated"}},
            {"tags.term": {"$ne":"No longer available"}},
            {"tags.term": {"$ne":"Unknown"}}
        ]
    }
    cttag_a = db_stocks.cttag_a.find(q)
    count = db_stocks.cttag_a.count(q)

    totalTrials = {}
    mgs_to_trialid = {}
    # for each tag record, collect composite keys of all tags
    ct = 0
    for cttag in cttag_a:
        ct+=1
        if ct % 1000 == 0:
            print(ct, 'tag records of', count)
        totalTrials[cttag['id']] = {}
        for tag in cttag["tags"]:
            if tag['filterType'] == 'include' and tag['suppress'] == False:
                # collect the dictionary of all tags for this trial, limit later
                compkey = tag["compositeKey"] + str(tag["priority"])
                totalTrials[cttag['id']][compkey] = True
                # find which medicalgroup this trial belongs to
                if tag["facet"] == "MEDICALGROUP" and tag["term"] in mgs:
                    if tag["term"] not in mgs_to_trialid:
                        mgs_to_trialid[tag["term"]] = [cttag['id']]
                    else:
                        mgs_to_trialid[tag["term"]].append(cttag['id'])

    # write trials to listed record(s)
    for mg in mgs_to_trialid:
        listed.update(
            {"medicalgroups":mg},
            {"$addToSet": {"trials": {"$each": mgs_to_trialid[mg]}}},
            upsert=False,
            multi=True
        )

    # apply mask for just the data I want for all historical data
    tags_I_Want = ["CONDITION", "PHASE", "DRUGCLASS"]
    priority_I_Want = ['include1', 'include2'] # include2 also
    headers = []
    # for each trial tag
    for trial in totalTrials:
        for tag in totalTrials[trial]:

            hasFacet = False
            hasPriority = False
            for f in tags_I_Want:
                if f in tag:
                    hasFacet = True
            for p in priority_I_Want:
                if p in tag:
                    hasPriority = True

            # if i want this tag, record it's compkey in headers
            if hasFacet and hasPriority:
                if tag not in headers:
                    headers.append(tag)

    # get the phase tags to the top of the headers, and sort the list
    phaseKeys = ["Phase 1PHASEinclude1", "Phase 2PHASEinclude1", "Phase 3PHASEinclude1", "Phase 4PHASEinclude1"]
    hnew = []
    for h in sorted(headers):
        if h not in phaseKeys:
            hnew.append(h)
    headers = phaseKeys + hnew
    print('total features: ', len(headers))

    # open a new tag table
    tagdata = db_stocks['tagdata']
    tagdata.remove({})

    # collect data based on these
    ct = 0
    for trial in totalTrials:
        ct+=1
        if ct % 1000 == 0:
            print(ct, 'trial ...', count)
        data = []
        for h in headers:
            if h in totalTrials[trial]:
                data.append(1)
            else:
                data.append(0)
        rec = {
            'id': trial,
            'tags': totalTrials[trial],
            'data': data,
            'lastupdated': today
        }
        tagdata.insert(rec, check_keys=False) #check_keys for '.' not all

    # do the same for historical tagdata
    for coll in db_stocks.collection_names():
        if 'tagdata-' in coll:
            ct=0
            for trial in db_stocks[coll].find({}):
                ct+=1
                if ct % 1000 == 0:
                    print(ct, 'trial historical of', count, coll)
                data = []
                for h in headers:
                    if h in trial['tags']:
                        data.append(1)
                    else:
                        data.append(0)
                # set the new data that is based on the new headers, but the old tags on the that trial
                db_stocks[coll].update(
                    {'id': trial['id']},
                    {'$set':{
                        'data': data,
                        'lastupdated': today
                    }}
                )


    print('ran tagcounts')

########################################################


def run_overrides():
    print('running ticker overrides')
    db = MongoClient("mongodb://localhost:27017").stocks
    # preferredtickers = np.genfromtxt('../resources/bpc_tickers.txt', dtype=np.str)
    # Some are duplicates or incorrect taggings like J & J Snack Foods Corp.
    badtickers = ["GCVRZ", "NOVT", "ICON", "VTNR", "JJSF", "CYCCP"]
    for tick in badtickers:
        licursor = db.listed.update(
            {"_symbol": tick},
            {"$set": {"suppress": True}}
        )

########################################################

def mgcalculate():
    print('running mg financials calculate')

    db = MongoClient("mongodb://localhost:27017").stocks
    q = {
        "medicalgroups.0":{"$exists":True},
        "trials.0":{"$exists":True},
        "marketcap": {"$exists":True},
        "marketcap": {"$gt":0},
        "suppress": {"$ne":True}
    }
    licursor = db.listed.find(q)
    count = db.listed.count(q)

    ct = 0
    beenseen = []
    for li in licursor:

        ct+=1
        if ct % 100 == 0:
            print(ct, 'mgcalculate ...', count)

        mgname = li['medicalgroups'][0] # first one??

        # check if duplicates coming through
        if mgname in beenseen:
            print('this one has duplicate: ', mgname, li['_symbol'], li['_name'] )
        beenseen = beenseen + li['medicalgroups']

        trials = li['trials']
        # use adjusted cap instead
        if 'operatingincome' in li and li["operatingincome"] and li["operatingincome"] > 0 and li["pipelineAdjustedMarketCap"] > 0:
            mc = li["pipelineAdjustedMarketCap"]
        else:
            mc = li["marketcap"]
        marketcapPerTrial = int( mc / len(trials) )

        # set on listed
        db['listed'].update(
            {"_id": li["_id"]},
            {"$set":{
                "marketcapPerTrial": marketcapPerTrial
            }}
        )

        # now go through each of these trials and write the marketcap as the Y target
        # for historical, keep the old Y
        for trial in trials:
            db['tagdata'].update(
                {"id": trial},
                {"$set":{
                    "marketcapPerTrial": marketcapPerTrial,
                    "medicalgroup": mgname,
                    "lastupdated": today
                }}
            )


########################################################

def backup():
    print('running backup')
    client = MongoClient('mongodb://localhost:27017')
    db_stocks = client.stocks
    # listed
    newcoll = db_stocks["listed-" + today]
    newcoll.remove({})
    for doc in db_stocks['listed'].find({}):
        newcoll.insert(doc, check_keys=False)
    # tagdata
    newcoll2 = db_stocks["tagdata-" + today]
    newcoll2.remove({})
    for doc in db_stocks['tagdata'].find({}):
        newcoll2.insert(doc, check_keys=False)

########################################################

if __name__ == "__main__":
    getlisted()
    mmdata()
    mgtagger()
    marketcap()
    tagcounts()
    run_overrides()
    mgcalculate()
    backup()

    # create a copy of the listeddb when this is ran? then process_data would
    # gather dbs from all dates. i should write the tagcounts to db as well so we can just back it all up by date.


########################################################
