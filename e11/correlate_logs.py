import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

line_re = re.compile("^(\\S+) - - \\[\\S+ [+-]\\d+\\] \"[A-Z]+ \\S+ HTTP/\\d\\.\\d\" \\d+ (\\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return [m.group(1), m.group(2)]
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    temp = log_lines.map(lambda k: line_to_row(k))
    return temp.filter(not_none)


def main():
    in_directory = sys.argv[1]
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    grouped = logs.groupBy('_1')

    df = grouped.agg(
        functions.count(logs['_1']).alias('x'),
        functions.sum(logs['_2']).alias('y')
    )

    six_vals = df.withColumn('x_2', df['x']**2)
    six_vals = six_vals.withColumn('y_2', df['y']**2)
    six_vals = six_vals.withColumn('xy', df['x'] * df['y'])
    six_vals = six_vals.withColumn('n', functions.lit(1))
    grouped = six_vals.groupBy()
    z = grouped.sum().first()
    numerator = (z['sum(n)'] * z['sum(xy)'] - z['sum(x)']*z['sum(y)'])
    a = math.sqrt(z['sum(n)'] * (z['sum(x_2)']) - z['sum(x)']**2)
    b = math.sqrt(z['sum(n)'] * (z['sum(y_2)']) - z['sum(y)']**2)
    denominator = a * b
    r = numerator / denominator
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    main()