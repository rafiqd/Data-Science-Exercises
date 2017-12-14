import sys
import string
import re
from pprint import pprint
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('word count').getOrCreate()

assert sys.version_info >= (3, 4)
assert spark.version >= '2.1'

def main():

    df = spark.read.text(sys.argv[1])
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation
    temp = df.select(functions.explode(functions.split(functions.lower(df['value']), wordbreak)).alias('word'))
    cleaned = temp.where(~(temp['word'].like(r'')))
    grouped = cleaned.groupBy(cleaned['word'])
    counts = grouped.agg(functions.count('word'))
    ordered = counts.orderBy(functions.desc('count(word)'), 'word')
    ordered.write.csv(sys.argv[2], mode='overwrite')

if __name__ == '__main__':
    main()