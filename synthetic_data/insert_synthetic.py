# https://docs.python.org/3/library/sqlite3.html

import sqlite3
import random

# ------ Helpers ------ #

s_corps = ['T-Mobile', 'Sound Transit', 'Koss', 'Subaru', 'PUD']
s_orgtype = ['communication', 'transportation', 'audio devices', 'cars', 'utilities']
s_numcust = [10000000000, None, 5280, 37068, 2]
s_numemp = [4380, 20, 18, None, 1234567890]

s_breachdate = ['2000-07-12 00:00:00', '2024-08-04 00:00:00', None, '2012-01-18 00:00:00', '2000-07-13 08:12:57']
s_numrec = [9025, 1689, 514575, 866126, 259333]
s_method = ['malware', 'insider threat', 'hacking', 'physical breach', 'accidental publishment']
s_numart = [None, None, None, 8, None]
s_sens = [0.5, 0.7, 0, 0.2, 0.9]

s_pubdate = s_breachdate
s_title = ['Data stolen from company', 'Cookie recipe', 'User info leaked', 'Cybersecurity attack succeeds', 'Best hiking trails']
s_lang = ['eng', 'spanish', 'italian', 'tagalog', 'portugese']
s_country = ['US', 'Mexico', 'Italy', 'Philippines', 'Brazil']
s_publisher = ['MSNBC', 'FOX', 'CNN', 'NY Times', 'Washington Post']
s_cat = ['data', 'cooking', 'consumer security', 'cybersecurity', 'recreation']
s_breachId = [None, 2, 4, 5, 8]

def read_file(filename):
  fd = open(filename, 'r')
  file = fd.read()
  fd.close()
  return file.split(';')

def rand_int(ceil):
  return random.randint(0, ceil)

# ------ Main ------ #

con = sqlite3.connect('synthetic.db')
cur = con.cursor()

# Create tables
creations = read_file('create_tables.sql')
for creation in creations:
  cur.execute(creation)

insertions = read_file('insert_synthetic.sql')
max_index = 4

# Insert synthetic corporations by index
ins_corp = insertions[0]
data_corp = []
for i in range(5):
  corp_entry = (s_corps[i], s_orgtype[i], s_numcust[i], s_numemp[i])
  data_corp.append(corp_entry)
cur.executemany(ins_corp, data_corp)

# Insert synthetic breaches with random permutations
ins_breach = insertions[1]
data_breach = []
for i in range(100):
  breach_entry = (s_corps[rand_int(max_index)],
                  s_breachdate[rand_int(max_index)],
                  s_numrec[rand_int(max_index)],
                  s_method[rand_int(max_index)],
                  s_numart[rand_int(max_index)],
                  s_sens[rand_int(max_index)])
  data_breach.append(breach_entry)
cur.executemany(ins_breach, data_breach)

# Insert synthetic news articles with random almost-permutations
ins_article = insertions[2]
data_article = []
for i in range(100):
  article_contents = rand_int(max_index)
  article_entry = (s_pubdate[rand_int(max_index)],
                  s_title[article_contents],
                  s_lang[rand_int(max_index)],
                  s_country[rand_int(max_index)],
                  s_publisher[rand_int(max_index)],
                  s_cat[article_contents],
                  s_breachId[rand_int(max_index)])
  data_article.append(article_entry)
cur.executemany(ins_article, data_article)

con.commit()
con.close()