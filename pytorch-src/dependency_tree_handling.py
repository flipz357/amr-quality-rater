import json
import logging
import spacy

logger = logging.getLogger("dep util")

def modify_nlp(nlp):
    def prevent_sentence_segmentation(doc):
        for token in doc:
	    # This will entirely disable spaCy's sentence detection
            token.is_sent_start = False
        return doc

    nlp.add_pipe(prevent_sentence_segmentation, name='prevent-sbd', before='parser')
    return None

def preprocess_and_clean_dep(sent, nlp, mode="DFS"):
    
    """function to preprocess a sentence and project it onto a multi-line string

    Args:
        sent (str): input sentence
        nlp: spacy nlp object
        mode (str): mode of linearization, default DFS

    Returns:
        string: a multi-line representation of the dependency tree
    """
    if mode == "BFS":
        #EXPERIMENTAL
        if False:
            return "ROOT "+"dummy"
        doc = nlp(sent)
        root = [token for token in doc if token.head == token][0]
        out = "ROOT "+root.lemma_+"\n"
        indentlevel=1
        visited = [False] * len(list(doc))
        queue = [root]
        visited[root.i] = True
        while queue:
            tmp = queue.pop(0)
            childfound=False
            for child in tmp.children:
                if visited[child.i] == False:
                    visited[child.i] = True
                    queue.append(child)
                    out+="+TAB+ " * indentlevel + ":" + child.dep_ + " " + child.lemma_ + "\n"
                    childfound=True
            if childfound:
                indentlevel+=1
        return out[:-1]
    
    elif mode == "DFS":

        #preprocess sent
        doc = nlp(sent)
        root = [token for token in doc if token.head == token][0]
        
        # init output first line
        out = "ROOT " + root.lemma_ + "\n"
        indentlevel=1

        # helper fucntion for tree traversal and collection
        def recursion(node, indentlevel):
            if not list(node.children):
                string = "+TAB+ " * indentlevel + ":" + node.dep_ + " " + node.lemma_ + "\n"
                return string
            else:
                string = "+TAB+ " * indentlevel + ":" + node.dep_
                compund_childs = list(sorted([node]
                    + [t for t in node.children if t.dep_ == "compound"], key=lambda t:t.i))
                for com in compund_childs:
                    string += " " + com.lemma_
                string += "\n"
                return string + "".join([recursion(ch, indentlevel + 1) 
                                         for ch in node.children if ch.dep_ != "compound"])
        
        # traverse tree from root and collect
        out+= "".join([recursion(ch, indentlevel) for ch in root.children])
        return out[:-1]



def add_2d_strings_dependency(jsondic, nlp, use_dependency=True, lookup_saver = True):
    
    """ analgously to add_2d_strings, we add the 2d strings for dependency
        
    Args:
        jsondic (dict): see above add_2d_strings
        nlp: spacy nlp object
        use_dependency: if true, we use the dependency tree if no,
            we use only the sentence
    Returns:
        None (the input dict will be modified)
    """

    # we keep a saver idx, since some of the sentences 
    # occur multiple times with different AMRs, so we don't need 
    # to preprocess them again
    saver = {}

    # iterate over data
    for i,key in enumerate(jsondic):
        
        # get the sentence
        snt = jsondic[key]["snt"]
        
        # check if we have seen the snt already, maybe preprocess
        if snt in saver and lookup_saver:
            jsondic[key]["dep2d"] = saver[snt]
            continue
        

        #preprocess dependency tree or jsut the sentence
        if use_dependency:
            tok = preprocess_and_clean_dep(snt, nlp)
            tok = "\n".join([t.replace("+TAB+ ", "", 1).strip() for t in tok.split("\n")])    
        else:
            tok = " ".join([t.lemma_ for t in nlp(snt)])

        # add preprocessed dep to data dict and save it
        jsondic[key]["dep2d"] = tok
        if lookup_saver:
            saver[snt] = tok
        if i % 1000 == 0:
            logger.info("deps loaded: {}/{}".format(i,len(jsondic)))
        
    return None


