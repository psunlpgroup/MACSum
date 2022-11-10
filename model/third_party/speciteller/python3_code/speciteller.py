
# coding: utf-8

import argparse
import os.path

import liblinear.liblinearutil as ll

import third_party.speciteller.python3_code.utils as utils
import sys
from third_party.speciteller.python3_code.features import Space
from third_party.speciteller.python3_code.generatefeatures import ModelNewText

RT = "/data/yfz5488/PCS/third_party/speciteller/"
if not os.path.exists(RT):
    RT = "/home/yfz5488/PCS/third_party/speciteller/"
BRNCLSTSPACEFILE = RT+"cotraining_models/brnclst1gram.space"
SHALLOWSCALEFILE = RT+"cotraining_models/shallow.scale"
SHALLOWMODELFILE = RT+"cotraining_models/shallow.model"
NEURALBRNSCALEFILE = RT+"cotraining_models/neuralbrn.scale"
NEURALBRNMODELFILE = RT+"cotraining_models/neuralbrn.model"

def initBrnSpace():
    s = Space(101)
    s.loadFromFile(BRNCLSTSPACEFILE)
    return s

def readScales(scalefile):
    scales = {}
    with open(scalefile) as f:
        for line in f:
            k,v = line.strip().split("\t")
            scales[int(k)] = float(v)
        f.close()
    return scales

brnclst = utils.readMetaOptimizeBrownCluster()
embeddings = utils.readMetaOptimizeEmbeddings()
brnspace = initBrnSpace()
scales_shallow = readScales(SHALLOWSCALEFILE)
scales_neuralbrn = readScales(NEURALBRNSCALEFILE)
model_shallow = ll.load_model(SHALLOWMODELFILE)
model_neuralbrn = ll.load_model(NEURALBRNMODELFILE)

def simpleScale(x, trainmaxes=None):
    maxes = trainmaxes if trainmaxes!=None else {}
    if trainmaxes == None:
        for itemd in x:
            for k,v in itemd.items():
                if k not in maxes or maxes[k] < abs(v): maxes[k] = abs(v)
    newx = []
    for itemd in x:
        newd = dict.fromkeys(itemd)
        for k,v in itemd.items():
            if k in maxes and maxes[k] != 0: newd[k] = (v+0.0)/maxes[k]
            else: newd[k] = 0.0
        newx.append(newd)
    return newx,maxes

def getFeatures(fin):
    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner.loadFromFile(fin)
    aligner.fShallow()
    aligner.fNeuralVec()
    aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    _,xw = aligner.transformWordRep()
    return y,xs,xw

def score(p_label, p_val):
    ret = []
    for l,prob in zip(p_label,p_val):
        m = max(prob)
        if l == 1: ret.append(1-m)
        else: ret.append(m)
    return ret

def predict(y,xs,xw):
    xs,_ = simpleScale(xs,scales_shallow)
    xw,_ = simpleScale(xw,scales_neuralbrn)
    p_label, p_acc, p_val = ll.predict(y,xs,model_shallow,'-q -b 1')
    ls_s = score(p_label,p_val)
    p_label, p_acc, p_val = ll.predict(y,xw,model_neuralbrn,'-q -b 1')
    ls_w = score(p_label,p_val)
    return [(x+y)/2 for x,y in zip(ls_s,ls_w)],ls_s,ls_w

def writeSpecificity(preds, outf):
    with open(outf,'w') as f:
        for x in preds:
            f.write("%f\n" % x)
        f.close()
    print("Output to "+outf+" done.")

def run(identifier, sentlist):
    ## main function to run speciteller and return predictions
    ## sentlist should be a list of sentence strings, tokenized;
    ## identifier is a string serving as the header of this sentlst
    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner.loadSentences(identifier, sentlist)
    aligner.fShallow()
    aligner.fNeuralVec()
    aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    _,xw = aligner.transformWordRep()
    preds_comb, preds_s, preds_w = predict(y,xs,xw)
    return preds_comb


########## Added by Yusen: yfz5488@psu.edu ####################
from nltk import sent_tokenize, word_tokenize
TMP_INFILE = "tmpin.txt"
def get_paragraph_specificity(text, metric='weighted'):
    text = sent_tokenize(text)
    text = [word_tokenize(x) for x in text]
    lengths = [len(x) for x in text]
    text = [' '.join(x) for x in text]

    with open(TMP_INFILE, 'w') as file:
        for line in text:
            file.write(line.strip() + '\n')
    y, xs, xw = getFeatures(TMP_INFILE)
    preds_comb, preds_s, preds_w = predict(y, xs, xw)
    if metric == 'weighted':
        result = sum(x*y for x, y in zip(preds_comb, lengths)) / sum(lengths)
    elif metric == 'avg':
        result = sum(preds_comb)/len(preds_comb)
    else:
        raise  NotImplementedError()
    return result
###################################################

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inputfile", help="input raw text file, one sentence per line, tokenized", default='../sents_test')
    argparser.add_argument("--outputfile", help="output file to save the specificity scores", default='../tmp.txt')
    argparser.add_argument("--write_all_preds", help="write predictions from individual models in addition to the overall one", action="store_true")
    # argparser.add_argument("--tokenize", help="tokenize input sentences?", required=True)
    sys.stderr.write("SPECITELLER: please make sure that your input sentences are WORD-TOKENIZED for better prediction.\n")
    args = argparser.parse_args()
    y,xs,xw = getFeatures(args.inputfile)
    preds_comb, preds_s, preds_w = predict(y,xs,xw)
    writeSpecificity(preds_comb,args.outputfile)
    if args.write_all_preds:
        writeSpecificity(preds_s,args.outputfile+".s")
        writeSpecificity(preds_w,args.outputfile+".w")

