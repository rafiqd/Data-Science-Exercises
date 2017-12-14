import sys
import os
import re
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('wikipeida popular').getOrCreate()

assert sys.version_info >= (3, 4)
assert spark.version >= '2.2'

schema = types.StructType([
    types.StructField('language', types.StringType(), False),
    types.StructField('title', types.StringType(), False),
    types.StructField('requests', types.IntegerType(), False),
    types.StructField('size', types.IntegerType(), False)
])


def convert(pathname):
    filename = os.path.basename(pathname)
    filename, ext = os.path.splitext(filename)
    match = re.search('([0-9]+-[0-9]{2})', filename)
    return match.group(0)

path_to_hour = functions.udf(convert, types.StringType())


def main(in_directory, out_directory):
    wikipeida_visits = spark.read.csv(
        in_directory,
        sep=' ',
        schema=schema).withColumn('filename', path_to_hour(functions.input_file_name()))

    wikipeida_visits = wikipeida_visits.filter(
        (wikipeida_visits['language'] == 'en') &
        (wikipeida_visits['title'] != 'Main_Page') &
        ~wikipeida_visits['title'].startswith('Special:')
    )

    grouped = wikipeida_visits.groupby(wikipeida_visits['filename'])
    max_score = grouped.agg(
        functions.max(wikipeida_visits['requests'])
    )
    joined_table = wikipeida_visits.join(max_score, on='filename')
    final = joined_table.filter(
        joined_table['requests'] == joined_table['max(requests)']
    ).orderBy([joined_table['filename'], joined_table['title']])

    output = final.select(['filename', 'title', 'requests'])
    output.write.csv(out_directory, mode='overwrite')

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)