import difflib
import sys
sys.path.append('..')

from fysom import fysom
from difflib import SequenceMatcher, _count_leading


#OURJUNK = lambda x: x in ('\n',' ','\t')
OURJUNK = None

class Overdiffer(object):
    def diff(self, text1, text2, split = '\n\n', depth=0):
        self.charjunk = OURJUNK

        if depth>3:
            return 'fail'
        pars1 = text1.strip().split(split)
        pars2 = text2.strip().split(split)

        matcher = SequenceMatcher(OURJUNK, pars1, pars2)
        for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
            if tag == 'replace':
                print '\n'.join([x.replace('\n', ' ') for x in
                        self._fancy_replace(
                            pars1, alo, ahi,
                            pars2, blo, bhi, .1)
                        ]
                        )
            else:
                print 'WOOT'+tag

    ##
    ## starting here was originally a copy from python2.6's difflib.py.
    ## adaptions include:
    ## * _fancy_replace cutoff is now a parameter
    ## * return a data structure describing the changes instead of yieding strings
    ##
    def _dump(self, tag, x, lo, hi):
        """Generate comparison results for a same-tagged range."""
        for i in xrange(lo, hi):
            yield '%s %s' % (tag, x[i])

    def _plain_replace(self, a, alo, ahi, b, blo, bhi):
        assert alo < ahi and blo < bhi
        # dump the shorter block first -- reduces the burden on short-term
        # memory if the blocks are of very different sizes
        if bhi - blo < ahi - alo:
            first  = self._dump('+', b, blo, bhi)
            second = self._dump('-', a, alo, ahi)
        else:
            first  = self._dump('-', a, alo, ahi)
            second = self._dump('+', b, blo, bhi)

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
                la, lb = ai2 - ai1, bj2 - bj1
                if tag == 'replace':
                    atags += '^' * la
                    btags += '^' * lb
                elif tag == 'delete':
                    atags += '-' * la
                elif tag == 'insert':
                    btags += '+' * lb
                elif tag == 'equal':
                    atags += ' ' * la
                    btags += ' ' * lb
                else:
                    raise ValueError, 'unknown tag %r' % (tag,)
            for line in self._qformat(aelt, belt, atags, btags):
                yield line
        else:
            # the synch pair is identical
            yield '  ' + aelt

        # pump out diffs from after the synch point
        for line in self._fancy_helper(a, best_i+1, ahi, b, best_j+1, bhi):
            yield line

    def _fancy_helper(self, a, alo, ahi, b, blo, bhi):
        g = []
        if alo < ahi:
            if blo < bhi:
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
            else:
                g = self._dump('-', a, alo, ahi)
        elif blo < bhi:
            g = self._dump('+', b, blo, bhi)

        for line in g:
            yield line

    def _qformat(self, aline, bline, atags, btags):
        r"""
        Format "?" output and deal with leading tabs.

        Example:

        >>> d = Differ()
        >>> results = d._qformat('\tabcDefghiJkl\n', '\t\tabcdefGhijkl\n',
        ...                      '  ^ ^  ^      ', '+  ^ ^  ^      ')
        >>> for line in results: print repr(line)
        ...
        '- \tabcDefghiJkl\n'
        '? \t ^ ^  ^\n'
        '+ \t\tabcdefGhijkl\n'
        '? \t  ^ ^  ^\n'
        """

        # Can hurt, but will probably help most of the time.
        common = min(_count_leading(aline, "\t"),
                     _count_leading(bline, "\t"))
        common = min(common, _count_leading(atags[:common], " "))
        atags = atags[common:].rstrip()
        btags = btags[common:].rstrip()

        yield "- " + aline
        if atags:
            yield "? %s%s\n" % ("\t" * common, atags)

        yield "+ " + bline
        if btags:
            yield "? %s%s\n" % ("\t" * common, btags)
    ##
    ## end of originally-copied difflib.py code
    ##



if __name__ == '__main__':
    import overdiff_test
    Overdiffer().diff(overdiff_test.text1, overdiff_test.text2)

