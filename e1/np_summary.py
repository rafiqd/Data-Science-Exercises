import numpy as np

def main():
    data = np.load('monthdata.npz')
    totals = data['totals']
    counts = data['counts']
    calculate_min(totals)
    calculate_avg_for_locations(totals, counts)
    calculate_avg_for_cities(totals, counts)
    total_precipitation_per_quarter(totals)


def total_precipitation_per_quarter(totals):
    print("Quarterly precipitation totals:")
    quarterly_totals = np.reshape(totals, (len(totals), 4, 3))
    sums = np.sum(quarterly_totals, axis=2)
    print(sums)


def calculate_avg_for_locations(totals, counts):
    print("Average precipitation in each month:")
    monthly_totals = np.sum(totals, axis=0)
    monthly_counts = np.sum(counts, axis=0)
    result = np.true_divide(monthly_totals, monthly_counts)
    print(result)


def calculate_avg_for_cities(totals, counts):
    print("Average precipitation in each city:")
    totals_sum = np.sum(totals, axis=1)
    counts_sum = np.sum(counts, axis=1)
    avg = np.true_divide(totals_sum, counts_sum)
    print(avg)


def calculate_min(totals):
    print("Row with lowest total precipitation:")
    totals_sum = np.sum(totals, axis=1)
    min = np.argmin(totals_sum)
    print(min)
    return min

if __name__ == '__main__':
    main()
