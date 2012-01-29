# README

Overdiff is a diff implementation based heavily on python 2.6's difflib
and even copying code from there.

Instead of returning the diff as text output, it returns data structures
describing changes. It also allows specification of a different threshold
for matching interline changes.

