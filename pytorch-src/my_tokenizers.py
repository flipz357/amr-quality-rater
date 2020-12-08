import logging
from collections import defaultdict
import re

logger = logging.getLogger("tokenizer")

class Tokenizer2D():

    """This object tokenizes a multi-line string to a 2D matrix"""

    def __init__(self
            , min_freq=5
            , lower=True
            , replace_num=True
            , splitter=" "):
        self.min_freq = min_freq
        self.lower = lower
        self.replace_num = replace_num
        self.splitter = splitter


    def maybe_lower(self,stringls):
        if self.lower:
            return [string.lower() for string in stringls]
        else:
            return stringls
    
    def get_reverse_vocab(self):
        return {v:k for k,v in self.vocab.items()}

    def fit(self, ls):
        
        """ takes a list with (multi-line) strings and computes vocabulary

        Args:
            ls (list): input list with strings

        Returns:
            None
        """

        ls=self.maybe_lower(ls)
        
        # init word count
        self.word_count = defaultdict(int)
        
        # iterate over strings
        for string in ls:
            #split multi line string
            tokenized=string.split("\n")
            # iterate over lines
            for line in tokenized:
                # split line
                line=line.split(self.splitter)
                # iterate over tokens in line and update word count
                for tok in line:
                    if self.replace_num:
                        if self.isint(tok) or self.isfloat(tok):
                            continue
                    self.word_count[tok]+=1

        # init vocabulary
        self.vocab = {"NOT_IN_VOCAB": 0
                ,"PADDING": 1
                ,"INTREPLACE": 2
                ,"FLOATREPLACE": 3}
        i=4

        # keep all tokens in vocab that occure >= min freq
        for t in self.word_count:
            if self.word_count[t] >= self.min_freq:
                self.vocab[t] = i
                i+=1
        self.reverse_vocab = self.get_reverse_vocab()
        return None
    
    def pad_crop(self, tmp, out_dim=(30,13)):
        
        """Funtion that pads and crops a 2d amr to the wished size.

        Args:
            tmp (list with list of ints): e.g. [[4],[5,6,7]]
            out_dim: desired out dimension 

        Returns:
            list with list of strings: e.g. if out_dim=(3,2), [[4,1],[5,6],[1,1]]
        """
        
        new_outer = []
        for j,ls in enumerate(tmp):
            new_inner = []
            for i,num in enumerate(ls):
                if i < out_dim[1]:
                    new_inner.append(num)
            while len(new_inner) < out_dim[1]:
                new_inner = new_inner + [1]
            if j < out_dim[0]:
                new_outer.append(new_inner)
        while len(new_outer) < out_dim[0]:
            #new_outer = [[1]*out_dim[1]]+new_outer
            new_outer = new_outer + [[1] * out_dim[1]]
        return new_outer
                
    def isint(self,s):
        if re.match(r"[1-9][0-9]*",s):
            return True
        return False 
   
    def isfloat(self,s):
        try: 
            float(s)
            return True
        except ValueError:
            return False 

    def apply(self, ls, out_dim=(30,13)):
        
        """Applies the tokenizer

        Args:
            ls (list with strings): the (multi-line) input strings/documents
            out_dim: the desired dimension of each 2d tokenized document

        Returns:
            list with lists of ints: tokenized documents
        """

        
        ls=self.maybe_lower(ls)
        out = []

        # we iterate over the input strings
        for string in ls:
            logger.debug("string to be tokenized: {}".format(string))
            tmp = []
            
            #we split the lines
            tokenized=string.split("\n")
            
            #we iterate over the lines
            for line in tokenized:
                tmp2 = []
                
                # and split each line into tokens
                line=line.split(self.splitter)
                
                # we iterate over the tokens
                for tok in line:

                    # if the token is in the vocab we insert its id
                    # if it's not in the vocab we check if it's a num or int
                    # and then replace it with a special idx, else we append 
                    # the "not in vocab" idx
                    if tok in self.vocab:
                        tmp2.append(self.vocab[tok])
                    else:
                        if self.replace_num:
                            if self.isint(tok):
                                tmp2.append(self.vocab["INTREPLACE"])
                            elif self.isfloat(tok):
                                tmp2.append(self.vocab["FLOATREPLACE"])
                            else:
                                tmp2.append(0)
                        else:
                            tmp2.append(0)
                tmp.append(tmp2)

            # pad and crop the processed list and append to output
            tmp = self.pad_crop(tmp, out_dim)
            out.append(tmp)
            logger.debug("string to be tokenized: {}".format(
                "\n".join([" ".join(
                    [self.reverse_vocab[idx] for idx in ls]) for ls in tmp])))

        return out

