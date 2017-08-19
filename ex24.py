print "Let's practice everything."
print 'You\'d need to know \'bout escapes with \\ that do \n newlines and \t tabs.'

poem = """
5 \tThe lovely world
6 with logic so firmly planted
7 cannot discern \n the needs of love
8 nor comprehend passion from intuition
9 and requires an explanation
10 \n\t\twhere there is none.
11 """
12
13 print "- - - - - - - - - - - - - - "
14 print poem
15 print "- - - - - - - - - - - - - - "
16
17
18 five = 10 - 2 + 3 - 6
19 print "This should be five: %s" % five
20
21 def secret_formula(started):
22 jelly_beans = started * 500
23 jars = jelly_beans / 1000
24 crates = jars / 100
25 return jelly_beans, jars, crates
26
27
28 start_point = 10000
29 beans, jars, crates = secret_formula(start_point)
30
31 print "With a starting point of: %d" % start_point
32 print "We'd have %d beans, %d jars, and %d crates." % (beans, jars, crates)
33
34 start_point = start_point / 10
35
36 print "We can also do that this way:"
37 print "We'd have %d beans, %d jars, and %d crates." % secret_formula(start_point)