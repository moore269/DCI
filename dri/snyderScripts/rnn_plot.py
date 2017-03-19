import os

#dataSets = ["facebook", "IMDB_5", "amazon_DVD_20000", "amazon_DVD_7500", "amazon_Music_64500", "amazon_Music_7500", "patents_computers_50attr"]
dataSets = ["patents_computers_50attr"]
#dataSets = ["facebook"]
queue = "csit"

for data in dataSets:
    submitStr = "qsub -q "+queue+" -v dataset='" + str(data) + "' rnn_exp_plot.sub"
    print(submitStr)
    os.system(submitStr)
