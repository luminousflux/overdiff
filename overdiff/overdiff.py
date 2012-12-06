import sys
import copy
from difflib import SequenceMatcher, _count_leading
from collections import defaultdict
from operator import itemgetter
import re

OURJUNK = None
TT = {'replace': '/','insert':'+','delete':'-'}

class Overdiffer(object):
    """Differ that returns structures instead of +/- marked strings and has customizable _fancy_replace.cutoff

       Based on 2.6 difflib.py algorithms

       >>> d = Overdiffer()
    """
    charjunk = OURJUNK

    def diff(self, pars1, pars2):
        """Diff input1 and input2, which are lists of strings, split at significant places (like paragraphs), yields differences

           :returns (+/-, astart, aend, bstart, bend, [aindexstart, aindexend, bindexstart, bindexend])
        """

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
        """Generate comparison results for a same-tagged range.
            
            >>> [x for x in Overdiffer()._dump(1,2,3,4,5,6)]
            [(1, 3, 4, 5, 6)]
        """
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

        >>> d = Overdiffer()
        >>> results = d._fancy_replace(['abcDefghiJkl\n'], 0, 1,
        ...                            ['abcdefGhijkl\n'], 0, 1)
        >>> [x for x in results]
        [('/', 0, 0, 0, 0, 3, 4, 3, 4), ('/', 0, 0, 0, 0, 6, 7, 6, 7), ('/', 0, 0, 0, 0, 9, 10, 9, 10)]
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


def find_splits(haystack, token, start, end):
    """Find occurences of tokens in haystack between start and end and return their indices
    
       >>> find_splits('asdf geh', ' ', 0, 7)
       [4]
       >>> find_splits('asdf geh', ' ', 5, 7)
       []
    """
    splits = []
    x = start-1
    while True:
        x = haystack.find(token, x+1)
        if x>=0 and x<=end:
            splits.append(x)
        else:
            break
    return splits


def split_at_token(haystack, hstart, hend, token, selections):
    """Splits selections at instances of token

    :param haystack   string to work in
    :param hstart     start of the workshpace
    :param hend       end of the workspace
    :param token      token at which to split selections
    :param selections selections to be split, in (start,end,weight,) format, sorted, non-overlapping

    :returns new selections

    there's a lot of inline list manipulation going on here, so this will be slow
    """

    sels = copy.copy(selections)
    tklen = len(token)

    splits = find_splits(haystack, token, hstart, hend)

    token_in_selection = splits

    while token_in_selection:
        token_in_selection = False
        for i, split in zip(xrange(0,len(splits)), splits):
            for sel in [(start,end,weight,) for (start,end,weight,) in sels if start<hend and end > hstart ]:
                start, end, weight = sel
                if split > start and split < end:
                    token_in_selection = True
                    sels.remove(sel)
                    # advance until token
                    while haystack[start:split].startswith(token) and start<split:
                        start+=tklen
                    if start<split:
                        sels.append((start, split, weight,))

                    split = split+tklen
                    # advance until end of token
                    while haystack[split:end].startswith(token) and split < end:
                        split += tklen
                    while haystack[end-tklen:end].endswith(token) and split < end:
                        # don't end at token
                        end -= tklen
                    if split<end:
                        sels.append((split, end, weight,))

    return sels


def expand_selection(haystack, hstart, hend, token, selections, expand_ratio = .5):
    """Expands selections between intervals, which are marked by tokens.

    :param haystack     a string in which to work
    :param hstart       start index of haystack to consider
    :param hstart       end index of haystack to consider
    :param token        tokens between which to expand (f.e. sentence boundary)
    :param selections   selections (start,end,weight,) to expand, sorted, non-overlapping
    :param expand_ratio threshold where to start expanding to whole space between tokens (f.e. 20% of sentence is selected? mark whole sentence!)

    >>> expand_selection('this is a sentence', 0, len('this is a sentence'), ' ', [(0,2,1)])
    [(0, 4, 1)]
    >>> expand_selection('this is a sentence', 0, len('this is a sentence'), ' ', [(5,6,1)])
    [(5, 7, 1)]
    """

    # there's no guarantee that incoming selections are split at the token.
    selections = split_at_token(haystack, hstart, hend, token, selections)

    sels = copy.copy(selections)

    splits = find_splits(haystack, token, hstart, hend)

    splits.insert(hstart, 0)
    splits.append(hend)

    for splitstart, splitend in zip(splits[0:len(splits)-1], splits[1:len(splits)]):
        splitsels = [(start,end,weight,) for (start,end,weight,) in sels if start<splitend and end > splitstart ]
        if haystack[splitstart:].startswith(token):
            splitstart += len(token)

        selected_weighted = sum([(end-start)*weight for (start,end,weight,) in splitsels])
        selected_num = sum([(end-start) for (start,end,weight,) in splitsels])

        if selected_num > (splitend-splitstart)*expand_ratio:
            for s in splitsels:
                sels.remove(s)
            # TODO: improve overestimation of weight here
            sels.append((splitstart, splitend, selected_weighted/(selected_num or 1),))
    return sels


def selection_to_s(haystack, selections, markdown=False):
    """return haystack with highlighted selected parts

       inline selections are marked with <ins> tags
       selected lines are prepended with .ins

       selections need to be in order
    """
    output = []
    cur = 0
    haystacklen = len(haystack)

    if markdown:
        selections = selections_split_markdown(haystack, selections)

    for sel in selections:
        start, end, weight = sel
        if cur > start:
            raise Exception('selection_to_s expects ordered input!')
        output.append(haystack[cur:start])

        block = start == 0 and end == len(haystack)
        if not block:
            output.append('<ins>%s</ins>' % haystack[start:end])
        else:
            output.append('.ins %s' % haystack[start:end].strip('\n'))
        cur = end
    output.append(haystack[cur:])
    return ''.join(output)

def _find_REs(REs, haystack):
    """ takes a list of regular expressions, returns a sorted list of (start,end) spans where either of them were found in haystack """
    matches = []
    for ble in REs:
        for m in re.finditer(ble, haystack):
            matches.append(m.span())
    matches.sort()
    return matches

def _connect_overlapping_selections(selections):
    newselections = []
    for selection in selections:
        if newselections and newselections[-1][1] >= selection[0]:
            if selection[1] > newselections[-1][0]: # exclude the case where selection is a subset of newselection[-1]
                t = list(newselections[-1])
                t[1] = selection[1]
                newselections[-1] = tuple(t)
        else:
            newselections.append(selection)
    return newselections

def _exclude_ranges_at_edges(selection, ranges):
    """ if a range is on the edge of a selection, exclude it

        >>> _exclude_ranges_at_edges((0,5,), [(0,1,), (1,2,), (4,6)])
        (1, 4)
    """
    start, end = selection[:2]

    otherstarts = [e for s,e in ranges if s<=start and e>start]
    if otherstarts:
        start = otherstarts[-1]
    otherends = [s for s,e in ranges if s<end and e>end]
    if otherends:
        end = otherends[0]
    return start, end

def _include_ranges_at_edges(selection, ranges):
    """ if a range is on the edge of a selection, include it

        ranges must be sorted.

        >>> _include_ranges_at_edges((4, 10,), [(0, 1,),(3,5),(-1,6,), (5,11)])
        (3, 11)
    """
    start, end = selection[:2]

    otherstarts = [s for s,e in ranges if s<start and e>start]
    if otherstarts:
        start = otherstarts[0]
    otherends = [e for s,e in ranges if s<end and e>end]
    if otherends:
        end = otherends[-1]
    return start, end


def selections_split_markdown(haystack, selections):
    import markdown

    untouchables = [r'\n\s*\*\s', r'\n\s*\+\s', r'\n\s*-\s', r'\n\s*\d+\.\s', r'^[ ]{0,3}\[([^\]]*)\]:\s*([^ ]*)[ ]*.*$', r'^<object.*>$', r'^<embed.*>$', r'^<iframe.*>$', r'\[imd\]', r'\[/imd\]', r'\|\|']

    impartibles = [markdown.inlinepatterns.LINK_RE,
            markdown.inlinepatterns.IMAGE_LINK_RE,
            markdown.inlinepatterns.IMAGE_REFERENCE_RE,]
    
    selections = _connect_overlapping_selections(selections)

    for x in untouchables[0:len(untouchables)]:
        untouchables.append(x.replace(r'\n',r'^'))
    string = haystack
    holes = _find_REs(untouchables, string)
    fills = _find_REs(impartibles, string)

    selectioncuts = []
    for selection in selections:
        start, end, weight = selection

        start, end = _exclude_ranges_at_edges(selection, holes)
        selection = (start, end, weight)
        start, end = _include_ranges_at_edges(selection, fills)
        selection = (start, end, weight)

        selectioncuts.append(start)
        for hole in [(s,e,) for (s,e,) in holes if s>=start and e<=end]:
            hs,he = hole
            selectioncuts.append(hs)
            selectioncuts.append(he)
        selectioncuts.append(end)

    selections = []
    while selectioncuts:
        if selectioncuts[1]-selectioncuts[0] > 0:
            selections.append(selectioncuts[0:2])
        selectioncuts = selectioncuts[2:]
    selections = [(s,e,1,) for (s,e,) in selections]

    selections = _connect_overlapping_selections(selections)

    return selections


def _each_with_index(collection):
    """ iterate over collection, return x, index """
    i = 0
    for x in collection:
        yield x,i
        i+=1

def _collect_nonnegative(function, x):
    """ call function w/ input, yield results while they're not negative"""
    while True:
        x = function(x)
        if x>=0:
            yield x
        else:
            break

def _ordered_pairs(collection):
    """ return ordered pairs of all items in collection

        >>> [x for x in _ordered_pairs([0,1,2,3])]
        [(0, 1), (1, 2), (2, 3)]
    """
    return zip(collection[:-1], collection[1:])

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

    sentenceendsfun = lambda x: paragraph2.find('.', x+1)           # TODO: should have used regex for tokens.
    sentenceends = list(_collect_nonnegative(sentenceendsfun, -1))
    sentenceends.insert(0,0)

    for x,y in _ordered_pairs(sentenceends):
        # TODO: not only expand between words, also expand between sentences.

        sels = expand_selection(paragraph2, x,y, ' ', sels)
        sels.sort(key=itemgetter(0))

    return sels

def overdiff(pars1, pars2):
    """return what changed in paragraphs as related to text2.
       this API _will_ be revised. """
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


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
