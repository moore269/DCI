import sys

start = int(sys.argv[1])
end = int(sys.argv[2])
buildStr = ""
for i in range(start, end+1):
    buildStr = buildStr + "qdel  "+str(i)+"; "
print(buildStr)
