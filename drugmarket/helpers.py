import pandas as pd
import numpy as np
import json
import io
import requests
from ftplib import FTP
import time
from decimal import Decimal
from pymongo import MongoClient

########################################################

def getlisted():
    print('running getlisted')
    # '''read in the listed files into db'''
    client = MongoClient("mongodb://localhost:27017")

    # create database stocks
    db_stocks = client.stocks
    # create collection and clear it out
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
            "Common Stock",
            "(Antigua/Barbudo)",
            "(Canada)",
            "Common Shares",
            "Holding Corporation",
            "Holding Company",
            "Holding Corp",
            "(Holding Company)"
            "Incorporated",
            " Inc",
            "Class A",
            "Ordinary Shares",
            "Depositary Shares",
            "Depositary Shares",
            " Ltd",
            ",",
            ".",
            "()"
        ]
        i['_name'] = i['Security Name']
        for _ng in nogood:
            i['_name'] = i['_name'].replace(_ng, '')
        i['_name'] = i['_name'].strip()

    listed.insert(records)

    print('ran listed')

########################################################

def mgtagger():
    print('running mgtagger')
    # use medicalgroups name and synonyms to tag the stock listings
    # https://www.quantshare.com/sa-426-6-ways-to-download-free-intraday-and-tick-data-for-the-us-stock-market
    # http://wern-ancheta.com/blog/2015/04/05/getting-started-with-the-yahoo-finance-api/
    # This downloads the current market cap
    # http://finance.yahoo.com/d/quotes.csv?s=GOOGL&f=j1
    print('running mg tagging ...')

    client = MongoClient("mongodb://localhost:27017")
    # to get this data, must buy license from http://api.molecularmatch.com
    molecularmatch = client.molecularmatch
    mgcursor = molecularmatch.medicalgroup.find({'exclude': False})

    db_stocks = client.stocks
    listed = db_stocks.listed

    # first unset all the medicalgroups
    listed.update(
        {},
        {'$unset': {'medicalgroups': True}},
        multi=True, upsert=False
    )

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

def phasecounts():
    print('running phasecounts')

    client = MongoClient("mongodb://localhost:27017")
    # to get this data, must buy license from http://api.molecularmatch.com
    molecularmatch = client.molecularmatch

    q = {
        "tags.facet": "MEDICALGROUP",
        "tags.term": "Recruiting"
    }
    cttag_a = molecularmatch.cttag_a.find(q)

    # create database stocks
    db_stocks = client.stocks

    # build an object in memory of each medicalgroups count of open trial phases
    mg_phase_count = {}
    mg_phase_condition_count = {}

    def countPhase(tags):
        pcounts = {
            'Phase 1': 0,
            'Phase 2': 0,
            'Phase 3': 0,
            'Phase 4': 0,
        }
        phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
        for i in tags:
            if i['facet'] == "PHASE" and i['suppress'] == False:
                for k in phases:
                    if i['term'] == k:
                        pcounts[k] += 1
                        break # or else a p1/p2 will get counted twice
                        # TODO: i think these are still getting counted twice
        return pcounts

    def totalTrialCount(pcounts):
        return pcounts['Phase 1'] + pcounts['Phase 2'] + pcounts['Phase 3'] + pcounts['Phase 4']

    # for each tag record
    for cttag in cttag_a:
        pcounts = countPhase(cttag["tags"])
        for tag in cttag["tags"]:
            if tag["facet"] == "MEDICALGROUP" and tag["suppress"] == False:
                # set the phase counts for all medicalgroups
                if not tag["term"] in mg_phase_count:
                    mg_phase_count[tag["term"]] = pcounts
                else:
                    for p in pcounts:
                        mg_phase_count[tag["term"]][p] += pcounts[p]

                # start recording what priority 1 conditions these trials are for
                # for tagc in cttag["tags"]:
                #     if tagc["facet"] == "CONDITION" and tagc["suppress"] == False and tagc["priority"] == 1:
                #         if not tag["term"] in mg_phase_condition_count:
                #             mg_phase_condition_count[tag["term"]] = tagc["term"]
                #         else:
                #             for p in pcounts:
                #                 mg_phase_condition_count[tag["term"]][p] += pcounts[p]

                # TODO: are asco abstracts tagging medicalgroup sponsors?

    # for each of these medicalgroup phase counts
    for i in mg_phase_count:
        # find where the stocks have this medicalgroup
        listedMG = list(db_stocks.listed.find({'medicalgroups': i}))
        if len(listedMG) == 1:
            # check to see if this trial update has more than the last (meaning it's a root tagging mg .. e.g. Pfizer tags, Pfizer, Inc. doesn't)
            update = True
            if 'phaseCounts' in listedMG[0]:
                totalNow = totalTrialCount(listedMG[0]['phaseCounts'])
                totalNew = totalTrialCount(mg_phase_count[i])
                if totalNew < totalNow:
                    update = False
            if update:
                db_stocks.listed.update(
                    {'_id': listedMG[0]['_id']},
                    {'$set': {'phaseCounts': {
                        'Phase 1': mg_phase_count[i]['Phase 1'],
                        'Phase 2': mg_phase_count[i]['Phase 2'],
                        'Phase 3': mg_phase_count[i]['Phase 3'],
                        'Phase 4': mg_phase_count[i]['Phase 4'],
                    }}}
                )
        elif len(listedMG) > 1:
            print('greater than 1 for ' + i)
            print(listedMG)
        elif len(listedMG) == 0:
            print('0 listed for ' + i)

    print('ran phasecounts')

########################################################

def conditioncounts():


########################################################

def marketcap():
    print('running marketcap')

    # This downloads the current market cap of the stock
    client = MongoClient("mongodb://localhost:27017")
    db_stocks = client.stocks
    listed = db_stocks.listed

    cursor = listed.find({"phaseCounts": {"$exists": True}})

    def text_to_num(text):
        d = {'M': 6, 'B': 9}
        if text[-1] in d:
            num, magnitude = text[:-1], text[-1]
            return int(Decimal(num) * 10 ** d[magnitude])
        else:
            return int(Decimal(text))

    for li in cursor:
        time.sleep(0.1)
        url = "https://api.iextrading.com/1.0/stock/" + li['_symbol'] + "/quote"
        with requests.Session() as s:
            download = s.get(url)
            content = json.loads(download.content.decode('utf-8'))
            db_stocks.listed.update(
                {'_id': li['_id']},
                {'$set': {"marketcap": content['marketCap']}}
            )

    print('ran marketcap')


########################################################
