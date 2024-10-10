import glob
import os


import socket
if 'darkwing' in socket.gethostname():
    os.environ['JAVA_HOME'] = sorted( glob.glob('/usr/lib/jvm/java-1.8.0-openjdk-*') )[-1] + '/jre'
    os.environ['SPARK_LOCAL_DIRS'] = '/scratch/' # '/m-ds1/bdata2/nobackup/galen_spark_scratch'
else:
    os.environ['SPARK_LOCAL_DIRS'] = '/tmp'


import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

print("Creating Spark session.")
configuation_properties = [
    ("spark.master","local[95]"),
    ("spark.ui.port","4050"),
    ("spark.executor.memory","750G"), # this may be ignored when running on a single machine
    ('spark.driver.memory',  '3T'),   # increased this from 2000g
    ('spark.driver.maxResultSize', '100G'), # updated from default of 1G
    ("spark.network.timeout",            "10000001"),
    ("spark.executor.heartbeatInterval", "10000000"),
#     ("spark.local.dir","/projects/bdata2/nobackup"), # updated to add nobackup
    ('spark.sql.autoBroadcastJoinThreshold','-1'), # override 8GB limit for broadcast joins, set no limit
]

conf = SparkConf().setAll( configuation_properties )

# create the context
sc = pyspark.SparkContext(conf=conf)
sc.setLogLevel('WARN') # can also be 'DEBUG'
spark = SparkSession.builder.getOrCreate()

print("Spark session created.")