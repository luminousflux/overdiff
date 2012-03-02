__version__ = '0.0.1'

__ALL__ = ['overdiff']

import sys
import copy
from difflib import SequenceMatcher, _count_leading
from collections import defaultdict
from operator import itemgetter

OURJUNK = None
TT = {'replace': '/','insert':'+','delete':'-'}

class Overdiffer(object):
    def diff(self, pars1, pars2):
        self.charjunk = OURJUNK

        matcher = SequenceMatcher(OURJUNK, pars1, pars2)
        for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
            if tag == 'replace':
                for x in self._fancy_replace(
                            pars1, alo, ahi,
                            pars2, blo, bhi, .1):
                    yield x
            elif tag != 'equal':
                yield (TT[tag], alo, ahi, blo, bhi)

    ##
    ## starting here was originally a copy from python2.6's difflib.py.
    ## adaptions include:
    ## * _fancy_replace cutoff is now a parameter
    ## * return a data structure describing the changes instead of yieding strings
    ##
    def _dump(self, tag, x, alo, ahi, blo, bhi):
        """Generate comparison results for a same-tagged range."""
        yield (tag, alo, ahi, blo, bhi)

    def _plain_replace(self, a, alo, ahi, b, blo, bhi):
        assert alo < ahi and blo < bhi
        # dump the shorter block first -- reduces the burden on short-term
        # memory if the blocks are of very different sizes
        if bhi - blo < ahi - alo:
            first  = self._dump('+', b, alo, ahi, blo, bhi)
            second = self._dump('-', a, alo, ahi, blo, bhi)
        else:
            first  = self._dump('-', a, alo, ahi, blo, bhi)
            second = self._dump('+', b, alo, ahi, blo, bhi)

        for g in first, second:
            for line in g:
                yield line

    def _fancy_replace(self, a, alo, ahi, b, blo, bhi, cutoff = 0.75):
        r"""
        When replacing one block of lines with another, search the blocks
        for *similar* lines; the best-matching pair (if any) is used as a
        synch point, and intraline difference marking is done on the
        similar pair. Lots of work, but often worth it.

        Example:

        >>> d = Differ()
        >>> results = d._fancy_replace(['abcDefghiJkl\n'], 0, 1,
        ...                            ['abcdefGhijkl\n'], 0, 1)
        >>> print ''.join(results),
        - abcDefghiJkl
        ?    ^  ^  ^
        + abcdefGhijkl
        ?    ^  ^  ^
        """


        # don't synch up unless the lines have a similarity score of at
        # least cutoff; best_ratio tracks the best score seen so far
        best_ratio = cutoff - 0.01
        cruncher = SequenceMatcher(self.charjunk)
        eqi, eqj = None, None   # 1st indices of equal lines (if any)

        # search for the pair that matches best without being identical
        # (identical lines must be junk lines, & we don't want to synch up
        # on junk -- unless we have to)
        for j in xrange(blo, bhi):
            bj = b[j]
            cruncher.set_seq2(bj)
            for i in xrange(alo, ahi):
                ai = a[i]
                if ai == bj:
                    if eqi is None:
                        eqi, eqj = i, j
                    continue
                cruncher.set_seq1(ai)
                # computing similarity is expensive, so use the quick
                # upper bounds first -- have seen this speed up messy
                # compares by a factor of 3.
                # note that ratio() is only expensive to compute the first
                # time it's called on a sequence pair; the expensive part
                # of the computation is cached by cruncher
                if cruncher.real_quick_ratio() > best_ratio and \
                      cruncher.quick_ratio() > best_ratio and \
                      cruncher.ratio() > best_ratio:
                    best_ratio, best_i, best_j = cruncher.ratio(), i, j
        if best_ratio < cutoff:
            # no non-identical "pretty close" pair
            if eqi is None:
                # no identical pair either -- treat it as a straight replace
                for line in self._plain_replace(a, alo, ahi, b, blo, bhi):
                    yield line
                return
            # no close pair, but an identical pair -- synch up on that
            best_i, best_j, best_ratio = eqi, eqj, 1.0
        else:
            # there's a close pair, so forget the identical pair (if any)
            eqi = None

        # a[best_i] very similar to b[best_j]; eqi is None iff they're not
        # identical


        # pump out diffs from before the synch point
        for line in self._fancy_helper(a, alo, best_i, b, blo, best_j):
            yield line

        # do intraline marking on the synch pair
        aelt, belt = a[best_i], b[best_j]
        if eqi is None:
            # pump out a '-', '?', '+', '?' quad for the synched lines
            atags = btags = ""
            cruncher.set_seqs(aelt, belt)
            for tag, ai1, ai2, bj1, bj2 in cruncher.get_opcodes():
                if tag == 'equal':
                    continue
                yield (TT[tag], best_i, best_i, best_j, best_j, ai1, ai2, bj1, bj2)
        else:
            # the synch pair is identical
            pass

        # pump out diffs from after the synch point
        for line in self._fancy_helper(a, best_i+1, ahi, b, best_j+1, bhi):
            yield line

    def _fancy_helper(self, a, alo, ahi, b, blo, bhi):
        g = []
        if alo < ahi:
            if blo < bhi:
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            else:
                g = self._dump('-', a, alo, ahi, blo, bhi)
        elif blo < bhi:
            g = self._dump('+', b, alo, ahi, blo, bhi)

        for line in g:
            yield line

    ##
    ## end of originally-copied difflib.py code
    ##


def split_at_token(haystack, hstart, hend, token, selections):
    # haystack = string
    # workspace = haystack[pstart:pend]
    # token = where to split selections
    # returns new selections

    sels = copy.copy(selections)
    splits = []

    token_in_selection = True
    tklen = len(token)

    x = hstart-1
    while True:
        x = haystack.find(token, x+1)
        if x>=0 and x<=hend:
            splits.append(x)
        else:
            break

    if not splits:
        token_in_selection = False


    while token_in_selection:
        token_in_selection = False
        for i, split in zip(xrange(0,len(splits)), splits):
            for sel in [(start,end,weight,) for (start,end,weight,) in sels if start<hend and end > hstart ]:
                start, end, weight = sel
                if split > start and split < end:
                    token_in_selection = True
                    sels.remove(sel)
                    while haystack[start:split].startswith(token) and start<split:
                        # don't start at token
                        start+=tklen
                    if start<split:
                        # don't start at token
                        sels.append((start, split, weight,))

                    split = split+tklen
                    while haystack[split:end].startswith(token) and split < end:
                        # don't start at token
                        split += tklen
                    while haystack[end-tklen:end].endswith(token) and split < end:
                        # don't end at token
                        end -= tklen
                    if split<end:
                        sels.append((split, end, weight,))

    return sels


def expand_selection(haystack, hstart, hend, token, selections, expand_ratio = .2):
    # token = tokens between which to expand
    # selections = selections to expand

    selections = split_at_token(haystack, hstart, hend, token, selections)

    sels = copy.copy(selections)

    splits = [hstart]
    x = hstart-1
    while True:
        x = haystack.find(token, x+1)
        if x>=0 and x<hend:
            splits.append(x)
        else:
            break

    splits.append(hend)

    for splitstart, splitend in zip(splits[0:len(splits)-1], splits[1:len(splits)]):
        splitsels = [(start,end,weight,) for (start,end,weight,) in sels if start<splitend and end > splitstart ]

        selected_weighted = sum([(end-start)*weight for (start,end,weight,) in splitsels])
        selected_num = sum([(end-start) for (start,end,weight,) in splitsels])

        if selected_num > (splitend-splitstart)*expand_ratio:
            for s in splitsels:
                sels.remove(s)
            sels.append((splitstart, splitend, selected_weighted/selected_num,))
                # this totally overestimates the weight here.
                # TODO: improve.
        splitsels = [(start,end,weight,) for (start,end,weight,) in sels if start<splitend and end > splitstart ]
    return sels


def selection_to_s(haystack, selections):
    output = []
    cur = 0
    haystacklen = len(haystack)
    for sel in selections:
        start, end, weight = sel
        if cur > start:
            raise Exception('selection_to_s expects ordered input!')
        output.append(haystack[cur:start])

        block = start == 0 and end == len(haystack)
        if not block:
            output.append('<ins>%s</ins>' % haystack[start:end])
        else:
            output.append('.ins %s' % haystack[start:end])
        cur = end
    output.append(haystack[cur:])
    return ''.join(output)

def _each_with_index(collection):
    i = 0
    for x in collection:
        yield x,i
        i+=1
def _collect_nonnegative(function, x):
    while True:
        x = function(x)
        if x>=0:
            yield x
        else:
            break

def _ordered_pairs(collection):
    for i,y in zip(xrange(0,len(collection)-1), xrange(1, len(collection))):
        yield collection[i], collection[y]

def overdiff_intraparagraph(paragraph2, diffs):
    sentences = [0]

    sels = []
    for x in diffs:
        if len(x)<=5:
            return [(0, len(paragraph2), 1)]
        alen = (x[6]-x[5]) or 1
        blen = (x[8]-x[7]) or 1 #  could be zero
        weight = (alen/blen) or 1
        sels.append( (x[7],x[8],weight,) )

    sentenceendsfun = lambda x: paragraph2.find('.', x+1)           # fuck fuck fuck fuck.
                                                                    # should have used regex
    sentenceends = list(_collect_nonnegative(sentenceendsfun, -1))
    sentenceends.insert(0,0)

    for x,y in _ordered_pairs(sentenceends):
        sels = expand_selection(paragraph2, x,y, '.', sels)
        sels.sort(key=itemgetter(0))
        sels = expand_selection(paragraph2, x,y, ' ', sels)
        sels.sort(key=itemgetter(0))
        sels = expand_selection(paragraph2, x,y, '.', sels)
        sels.sort(key=itemgetter(0))

    return sels

def overdiff(pars1, pars2):
    """return what changed in paragraphs as related to text2.
       this API _will_ be revised. """
    #t1 = text1.strip().split(token)
    #t2 = text2.strip().split(token)
    t1 = pars1
    t2 = pars2

    ds = Overdiffer().diff(t1, t2)

    related_diffs = defaultdict(list)

    for d in ds:
        tag, alo, ahi, blo, bhi = d[:5]
        for x in xrange(blo, bhi+1):
            related_diffs[x].append(d)

    for line, index in _each_with_index(t2):
        if related_diffs[index]:
            yield overdiff_intraparagraph(line, related_diffs[index])
        else:
            yield []


def overdiff_and_highlight(text1, text2, token='\n\n'):
    pars1 = text1.split(token)
    pars2 = text2.split(token)

    output = []

    diffs = list(overdiff(pars1,pars2))
    for i in xrange(0,len(pars2)):
        output.append(selection_to_s(pars2[i], diffs[i]))

    return token.join(output)


