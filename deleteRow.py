import csv

#remove_from = 303
#remove_to = 310

with open("data/trainingData.csv", "r") as fp_in, open("data/updated_trainingData.csv", "w") as fp_out:
    reader = csv.reader(fp_in, delimiter=",")
    writer = csv.writer(fp_out, delimiter=",")
    for row in reader:
        new_row = [col for idx, col in enumerate(row) if idx not in (0, 10,14,16,17,18,60,61,62,63,64,65,302,303,304,305,306,307,308,309)]
        #del row[remove_from:remove_to]
        writer.writerow(new_row)
