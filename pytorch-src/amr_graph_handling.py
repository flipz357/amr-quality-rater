import logging
import re

logger = logging.getLogger("amr utils")


def add_2d_strings(jsondic, wikiopt="keep", senseopt="keep", reentrancyopt="rvn"):

    """ function that adds the 2d representation of AMR to the data dict

    Args:
        jsondic (dict): data dict, {0: {"amr": "(x / say ....)"}...}
        wikiopt (str) : see function preprocess_and_clean
        senseopt (str) : see function preprocess_and_clean
        reentracyopt (str) : see function preprocess_and_clean
    
    Returns:
        None (the input dict will be modified)
    """
    
    # we iterate over the data examples
    for i,key in enumerate(jsondic):

        #we get the raw amr
        amr = jsondic[key]["amr"]
        
        #we clean and preprocess the amr to multi-line string
        tok = preprocess_and_clean_amr(amr
                , wikiopt=wikiopt
                , senseopt=senseopt
                , reentrancyopt=reentrancyopt
                )

        # the first indent level can be removed
        tok = "\n".join([t.replace("+TAB+ ", "", 1).strip() for t in tok.split("\n")])    
        
        # we save the new amr representation in our dict
        jsondic[key]["amr2d"] = tok
        
        # print some info
        if i % 1000 == 0:
            logger.info("amrs loaded: {}/{}".format(
                i, len(jsondic)))
    return None


def get_var_dict(amrlines):
    d = {}
    for line in amrlines:
        spl = line.split()
        for j,t in enumerate(spl[:-1]):
            if t == "/":
                d[spl[j-1]] = spl[j+1]
    return d

def handle_vars(amrlines, method="concept"):
    var_dict = get_var_dict(amrlines)
    if method == "concept":
        for i, line in enumerate(amrlines):
            spl = line.split()
            for j,t in enumerate(spl):
                if t in var_dict:
                    jspl = " ".join(spl).replace(" " + var_dict[t], " ")
                    if ":" in jspl:
                        jspl = jspl.replace(" " + t, " xXxXx")
                    else:
                        jspl = jspl.replace(t, "xXxXx")
                    jspl = jspl.replace("/", "").replace("xXxXx", var_dict[t])
                    amrlines[i] = jspl
                    break
    if method == "rvn":
        varptr = {}
        k=0
        for i,line in enumerate(amrlines):
            spl = line.split()
            for j,t in enumerate(spl):
                if t in var_dict:
                    if t not in varptr:
                        varptr[t]=""
                    else:
                        if "<" not in varptr[t]:
                            varptr[t] = "<"+str(k)+">"
                            k+=1
                    break
        for i,line in enumerate(amrlines):
            spl = line.split()
            for j,t in enumerate(spl):
                if t in var_dict:
                    if "/" not in spl and t == var_dict[t]:
                        jspl = " ".join(spl)
                    else:
                        jspl = " ".join(spl).replace(" "+var_dict[t], " ", 1)
                    if ":" in jspl:
                        jspl = jspl.replace(" " + t, " xXxXx")
                    else:
                        jspl = jspl.replace(t,"xXxXx")
                    jspl = jspl.replace("/","").replace("xXxXx", varptr[t] + " " + var_dict[t])
                    amrlines[i] = jspl
                    break
    return None


def simplify_names(amrlines):
    delis = []
    for i in range(len(amrlines)):
        if ":name" in amrlines[i]:
            ops =""
            for j, l in enumerate(amrlines[i+1:]):
                if l.count("+TAB+") < amrlines[i].count("+TAB+"):
                    break
                if re.match(r".*:op[0-9]+ ", l):
                    ops+=re.split(r":op[0-9]+ ", l)[1]+" "
                    delis.append(i+j+1)
                else:
                    break
            amrlines[i] = amrlines[i].replace(" name", " "+ops.strip())
    new = [amrline for i, amrline in enumerate(amrlines) if i not in delis]
    return new

def preprocess_and_clean_amr(amrstring
        , wikiopt="keep"
        , senseopt="keep"
        , reentrancyopt="rvn"):
    
    """ AMR formatting and cleaning

    This function takes a string AMR e.g., (j / jump-01 :ARG0 (f / frog)) \
    and returns a formated multiline string, e.g.
    jump -01
    +TAB+ ARG0 frog

    Args:
        amrsting (str): input AMR
        senseopt (str): how to handle senses, default jump-01 --> jump -01
        renentrancyopt (str): how to handle reentrancies, currently its 
            loss-less variable replacement with pointers according to RVN 17,
            e.g., (s / scratch :ARG0 (c / cat) :ARG1 c) 
                ---> scratch :ARG0 <1> cat :ARG1 <1> cat
            e.g., (s / scratch :ARG0 (c / cat) :ARG1 (o / cat)) 
                ---> scratch :ARG0 <1> cat :ARG1 <2> cat
        wikiopt (str): how to handle wiki links, default: keep and tokenize
    Returns:
        a formatted clean multiline AMR string with indents (+TAB+)
    """

    logger.debug("amr before simple cleaning: {}".format(amrstring))
    amrstring = amrstring.replace("\'", "\"")
    amrstring = amrstring.replace("\n", " ")
    amrstring = amrstring.replace("\t", " ")
    amrstring = amrstring.replace("_(","WIKILBR").replace(")\"", "WIKIRBR\"")
    amrstring = amrstring.replace("(", "( ")
    amrstring = amrstring.replace(")", " )")
    amrstring_toks = " ".join(amrstring.split())
    logger.debug("amr after simple cleaning: {}".format(amrstring))
    
    amrstring_toks = amrstring_toks.split(" ")
    toks = []
    lbr = 0
    rbr = 0
    rbr_before_rel= 0
    lbr_before_rel= 0
    tmpline = ""
    for i, tok in enumerate(amrstring_toks):
        if tok == "(":
            lbr+=1
            lbr_before_rel += 1
            continue
        if tok == ")":
            rbr+=1
            rbr_before_rel +=1
            continue
        if tok[0] == ":":
            tmpline = "+TAB+" * (lbr - rbr + rbr_before_rel - lbr_before_rel) + " " + tmpline
            tmpline = tmpline.strip()
            toks.append(tmpline)
            tmpline = tok+" "
            rbr_before_rel = 0
            lbr_before_rel = 0
        else:
            tmpline += tok+" "
        if not any([":" in t for t in amrstring_toks[i:]]):
            tmpline += " ".join([t for t in amrstring_toks[i+1:] if t != ")"])
            break
        
    tmpline = "+TAB+" * (lbr - rbr + rbr_before_rel - lbr_before_rel) + " " + tmpline
    tmpline = tmpline.strip()
    toks.append(tmpline)
    handle_vars(toks, method=reentrancyopt)
    toks = simplify_names(toks)
    toks = [t.replace("\"","").replace("+TAB+", "+TAB+ ") for t in toks]
    toks = [" ".join(t.split()) for t in toks]
    toks = [t.replace(":polarity -", ":polarity neg") for t in toks]
    toks = [t.replace("-0", "HYPHENZERO") for t in toks]
    toks = [t.replace("-9", "HYPHENNINE") for t in toks]
    toks = [t.replace("HYPHENZERO", " -0") for t in toks]
    toks = [t.replace("HYPHENNINE", " -9") for t in toks]
    if senseopt == "remove":
        toks = [re.sub(r"-[0-9]+", "", t) for t in toks ]
    toks = [t.replace("WIKILBR", " ") for t in toks]
    toks = [t.replace("WIKIRBR", " ") for t in toks]
    toks = [t.replace("_", " ") for t in toks]
    logger.debug("amr after advanced cleaning/formatting: {}".format("\n".join(toks)))
    if wikiopt == "remove":
        toks = [t for t in toks if ":wiki" not in t]
    if wikiopt == "one":
        wikis = [t for t in toks if ":wiki" in t]
        toks = [t for t in toks if ":wiki" not in t]
        if wikis:
            toks.append(":haswikis")
    if wikiopt == "append":
        wikis = [t.replace("+TAB+ ", "") for t in toks if ":wiki" in t]
        toks = [t for t in toks if ":wiki" not in t]
        if wikis:
            toks = toks + wikis
    toks = [t.strip() for t in toks]
    logger.debug("amr after advanced cleaning/formatting: {}".format("\n".join(toks)))
    return "\n".join(toks)

