import pandas as pd

def main():
    totals = pd.read_csv('totals.csv').set_index(keys=['name'])
    counts = pd.read_csv('counts.csv').set_index(keys=['name'])
    calculate_min(totals)
    calculate_avg_for_locations(totals, counts)
    calculate_avg_for_cities(totals, counts)


def calculate_avg_for_locations(totals, counts):
    print("Average precipitation in each month:")
    totals_sum = totals.sum(axis=0)
    counts_sum = counts.sum(axis=0)
    average = totals_sum / counts_sum
    print(average)

def calculate_avg_for_cities(totals, counts):
    print("Average precipitation in each city:")
    totals_sum = totals.sum(axis=1)
    counts_sum = counts.sum(axis=1)
    average = totals_sum / counts_sum
    print(average)

def calculate_min(totals):
    summed_totals = totals.sum(axis=1)
    print("City with lowest total precipitation:")
    print(summed_totals.idxmin(axis=0))

if __name__ == '__main__':
    main()