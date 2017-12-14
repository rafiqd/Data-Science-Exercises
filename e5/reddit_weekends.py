import pandas as pd
import numpy as np
import sys
import gzip
import scipy.stats

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)


def main(filename):
    reddit_fh = gzip.open(filename, 'rt', encoding='utf-8')
    reddit = pd.read_json(reddit_fh, lines=True)
    dates = pd.Series([2013, 2012])
    filtered = reddit[(reddit['date'].dt.year.isin(dates)) & (reddit['subreddit'] == 'canada')]
    weekend_filter = filtered['date'].dt.weekday.isin([5, 6])
    weekends = filtered.ix[weekend_filter]
    weekdays = filtered.ix[~weekend_filter]
    initial_statistic, initial_pvalue = scipy.stats.ttest_ind(weekends['comment_count'], weekdays['comment_count'])

    initial_weekend_normal_test = scipy.stats.normaltest(weekends['comment_count'])
    initial_weekday_normal_test = scipy.stats.normaltest(weekdays['comment_count'])
    variances = scipy.stats.levene(weekends['comment_count'], weekdays['comment_count'])

    transformed_weekend_normal_test = scipy.stats.normaltest(np.sqrt(weekends['comment_count']))
    transformed_weekday_normal_test = scipy.stats.normaltest(np.sqrt(weekdays['comment_count']))
    transformed_variances = scipy.stats.levene(np.sqrt(weekends['comment_count']), np.sqrt(weekdays['comment_count']))

    def get_iso_week(df):
        year, week, day = df['date'].isocalendar()
        return "%s%s" % (str(year), str(week))

    weekdays.is_copy = False
    weekends.is_copy = False
    weekends['weekid'] = weekends.apply(get_iso_week, axis=1)
    weekdays['weekid'] = weekdays.apply(get_iso_week, axis=1)
    weekends_group_means = weekends.groupby(['weekid'])['comment_count'].mean()
    weekdays_group_means = weekdays.groupby(['weekid'])['comment_count'].mean()
    print(weekdays_group_means)
    print(weekends_group_means)

    weekly_weekend_normal_test = scipy.stats.normaltest(weekends_group_means)
    weekly_weekday_normal_test = scipy.stats.normaltest(weekdays_group_means)
    weekly_variances = scipy.stats.levene(weekends_group_means, weekdays_group_means)
    weekly_statistic, weekly_pvalue = scipy.stats.ttest_ind(weekends_group_means, weekdays_group_means)

    utest_stat, utest_pval = scipy.stats.mannwhitneyu(weekends['comment_count'], weekdays['comment_count'])

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_pvalue,
        initial_weekday_normality_p=initial_weekday_normal_test[1],
        initial_weekend_normality_p=initial_weekend_normal_test[1],
        initial_levene_p=variances[1],
        transformed_weekday_normality_p=transformed_weekday_normal_test[1],
        transformed_weekend_normality_p=transformed_weekend_normal_test[1],
        transformed_levene_p=transformed_variances[1],
        weekly_weekday_normality_p=weekly_weekday_normal_test[1],
        weekly_weekend_normality_p=weekly_weekend_normal_test[1],
        weekly_levene_p=weekly_variances[1],
        weekly_ttest_p=weekly_pvalue,
        utest_p=utest_pval,
    ))

if __name__ == "__main__":
    main(sys.argv[1])
