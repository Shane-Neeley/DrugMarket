# -*- coding: utf-8 -*-
'''
Model Hypothesis: The value of a company without revenue from an approved drug is related
to the quality and number of it's clinical trials. So calculate a dollar value per trial for
each company. This assumption is more accurate as the drugs in development. Big companies with high
Market Cap (MC) are going after big targets w/ important trials. Therefore small companies with
similar trials should expect a similar large payout. To remove revenue from MC,
subtract some revenue multiplier (5 years).

DISCLAIMER: these assumptions are comletely foolish. If this is the kind of the quants on wall street do,
then they are insane.

$ / trial = MC / Ntrials

Create a neural network to learn to predict dollar value per trial:
X = (Ntrials, [tags, nlp, paper's tags]) ... each sample is a trial and the data we can gather on it.
Y = MC / N trials(company) ... the target is the dollar value per trial.

Then pass a company's pipeline forward through this network to predict a learned pipeline value.
Sort stocks by pipeline value for investment decisions.
'''

from pymongo import MongoClient
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tabulate import tabulate
import copy
import json

# from helpers import getlisted, mgtagger, phasecounts, marketcap
# getlisted()
# mgtagger()
# phasecounts()
# marketcap()

def runCore():


if __name__ == "__main__":
    runCore()
