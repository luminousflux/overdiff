import sys
sys.path.append('..')
import overdiff
import unittest

text1 = """
Hello there.

This sentence will be removed from here.

This is a long paragraph explaining an issue in a very detailed fashion. It explores
if line breaks are a major problem for us and gives us an opportunity to check how
line breaks affect diff performance.
It also includes some wyrd spllng mstks tht ought to be rectified.

This is a yet to be written conclusion, after which the text ends.
"""


text2 = """
This sentence was added to the top.

Hello there.

This is a long paragraph explaining an issue in a very detailed fashion. It explores
if line breaks are a major problem for us and gives us an opportunity to check how
line breaks affect diff performance. Even more detail is added in the middle to check
how that affects the output. Does this have the desired effect?
It also includes some weird spelling mistakes that ought to be rectified.

This is a pretty poorly- written conclusion, after which the text is supposed to end. for now.
"""

haystack = "asdfasdfasdfasdfasdfa sdfasdfas dfasdf   asdfasdfasdf asdf"


class OverdiffTest(unittest.TestCase):
    def testSplit(self):
        start = 0
        end = len(haystack)
        selection = [(start,end,1,)]

        expected_selections = set([(0,21,1,),(22,31,1,),
            (32,38,1,),(41,53,1,),(54,58,1,)])

        selections = overdiff.split_at_token(haystack,start,end,' ', selection)
        selections = set(selections)

        self.assertFalse(self._selections_are_nonoverlapping(selections),
                selections)
        self.assertEquals(expected_selections,
                selections,
                'same: %s / difference: %s' %
                    (expected_selections.intersection(selections),
                     expected_selections.symmetric_difference(selections)
                    ))

    def testExpand(self):

        start = 0
        end = len(haystack)

        selection = [(10, 15, 1,), (16, 18, 1),]

        sels = overdiff.expand_selection(haystack, start, end, ' ', selection)

    def _selections_are_nonoverlapping(self, selections):
        for x in selections:
            for y in selections:
                if x!=y:
                    s1,e1,w1 = x
                    s2,e2,w2 = y
                    if s1<s2:
                        if e1>=s2:
                            return '%s and %s overlap' % (x,y,)
                    if e1<e2:
                        if s2<=e1:
                            return '%s and %s overlap' % (x,y,)
        return None

    def testDiff(self):
        diffs = list(overdiff.overdiff(text1, text2))


if __name__ == '__main__':
    unittest.main()

