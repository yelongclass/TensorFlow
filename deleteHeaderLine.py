with open("data/trainingData.csv",'r') as f:
    with open("data/updated_trainingData.csv",'w') as f1:
        f.next() # skip header line
        for line in f:
            f1.write(line)
