import sys
import pandas as pd
from scipy import stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    searches = pd.read_json(searchdata_file, orient='records', lines=True)

    a_zero_searches = searches[(searches['uid'] % 2 == 0) & (searches['search_count'] == 0)]
    b_zero_searches = searches[(searches['uid'] % 2 != 0) & (searches['search_count'] == 0)]
    a_searches = searches[(searches['uid'] % 2 == 0) & (searches['search_count'] != 0)]
    b_searches = searches[(searches['uid'] % 2 != 0) & (searches['search_count'] != 0)]

    contingency_array = [
        [a_zero_searches.shape[0], a_searches.shape[0]],
        [b_zero_searches.shape[0], b_searches.shape[0]]
    ]

    chi2, p, dof, expected = stats.chi2_contingency(contingency_array)
    a_searches = searches[(searches['uid'] % 2 == 0)]
    b_searches = searches[(searches['uid'] % 2 != 0)]
    stat, pvalue = stats.mannwhitneyu(
        a_searches['search_count'],
        b_searches['search_count']
    )

    i_a_zero_searches = searches[(searches['uid'] % 2 == 0) & (searches['search_count'] == 0) & (searches['is_instructor'])]
    i_b_zero_searches = searches[(searches['uid'] % 2 != 0) & (searches['search_count'] == 0) & (searches['is_instructor'])]
    i_a_searches = searches[(searches['uid'] % 2 == 0) & (searches['search_count'] != 0) & (searches['is_instructor'])]
    i_b_searches = searches[(searches['uid'] % 2 != 0) & (searches['search_count'] != 0) & (searches['is_instructor'])]

    instructor_contingency_array = [
        [i_a_zero_searches.shape[0], i_a_searches.shape[0]],
        [i_b_zero_searches.shape[0], i_b_searches.shape[0]]
    ]

    chi2, instructor_p, dof, expected = stats.chi2_contingency(instructor_contingency_array)
    instructor_a_searches = searches[(searches['uid'] % 2 == 0) & (searches['is_instructor'])]
    instructor_b_searches = searches[(searches['uid'] % 2 != 0) & (searches['is_instructor'])]
    stat, instructor_pvalue = stats.mannwhitneyu(
        instructor_a_searches['search_count'],
        instructor_b_searches['search_count']
    )

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p,
        more_searches_p=pvalue,
        more_instr_p=instructor_p,
        more_instr_searches_p=instructor_pvalue,
    ))


if __name__ == '__main__':
    main()
