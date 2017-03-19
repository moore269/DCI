import sys

def generateCombine(input_name):
    f = open(input_name+".attr")
    attrs = {}
    for line in f:
        fields = line.replace("\n", "").split("::")
        nodeID = int(fields[0])

        attr = map(float, fields[1:])
        attrs[nodeID]=attr

    f = open(input_name+"N2V.attr")
    newlen = 0
    for i, line in enumerate(f):
        if i==0:
            continue
        fields = line.replace("\n", "").split()
        nodeID = int(fields[0])
        attr = map(float, fields[1:])
        attrs[nodeID]=attrs[nodeID] + attr
        newlen = len(attrs[nodeID])

    for key in attrs:
        strBuilder = str(key)
        if len(attrs[key])==newlen:
            for attrval in attrs[key]:
                strBuilder = strBuilder + "::"+str(attrval)
            print(strBuilder)

generateCombine(sys.argv[1])
