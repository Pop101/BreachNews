import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:20,.2f}'.format
to_write = ''

# Year, Company Name, Type of Breach, Records Compromised, Industry,
# Financial Loss, Impact Level, Human Error Factor, Human Error Factor Code,
# Mitigation Measures
huz = pd.read_csv('breaches_huz1020.csv', skiprows=0)

# ID, Entity, Year, Records, Organization type, Method, Sources
dev = pd.read_csv('breaches_devastator.csv', skiprows=0)
dev = dev.rename(columns={'Unnamed: 0': 'ID'})
dev = dev[dev['Year'].astype(str).str.isdigit()]
dev['Year'] = pd.to_numeric(dev['Year'])
dev = dev[dev['Records'].astype(str).str.isdigit()]
dev['Records'] = pd.to_numeric(dev['Records'])

# organisation, alternative name, records lost, year, date, story, sector,
# method, interesting story, data sensitivity,displayed records, , source name,
# 1st source link, 2nd source link, ID
info = pd.read_csv('breaches_information.csv', skiprows=range(1, 26))
info = info.rename(columns={'year   ': 'year'})
info['records lost'] = info['records lost'].str.replace(',', '')
info['records lost'] = pd.to_numeric(info['records lost'])

# Compare distributions/summaries of records leaked
to_write += '!!! Records leaked comparisons\n'
to_write += 'Huz dataset\n'
to_write = to_write + huz['Records Compromised'].describe().to_string() + '\n\n'
to_write += 'Devastator dataset\n'
to_write = to_write + dev['Records'].describe().to_string() + '\n\n'
to_write += 'Info dataset\n'
to_write = to_write + info['records lost'].describe().to_string() + '\n\n'

bins = 50
lim = (0, 1000000000)
plt.hist(huz['Records Compromised'],
         bins=bins,
         range=lim,
         alpha=0.5,
         label='Huz',
         color='blue',
         density=True)
sns.kdeplot(huz['Records Compromised'], color='blue', clip=lim, bw_method=0.6)
plt.hist(dev['Records'],
         bins=bins,
         range=lim,
         alpha=0.5,
         label='Dev',
         color='red',
         density=True)
sns.kdeplot(dev['Records'], color='red', clip=lim, bw_method=0.6)
plt.hist(info['records lost'],
         bins=bins,
         range=lim,
         alpha=0.5,
         label='Info',
         color='green',
         density=True)
sns.kdeplot(info['records lost'], color='green', clip=lim, bw_method=0.6)
plt.legend(loc='upper right')
plt.title('Records lost distribution')
plt.yscale('log')
plt.show()

# Compare distributions/summaries of sectors
dev_sec = dev['Organization type'].mask(dev['Organization type'] == 'web', 'Technology')
dev_sec = dev_sec.mask(dev_sec == 'tech', 'Technology')
dev_sec = dev_sec.mask(dev_sec == 'social media', 'Technology')
dev_sec = dev_sec.mask(dev_sec == 'web, tech', 'Technology')
dev_sec = dev_sec.mask(dev_sec == 'tech, web', 'Technology')
dev_sec = dev_sec.mask(dev_sec == 'QR code payment', 'Technology')
dev_sec = dev_sec.mask(dev_sec == 'healthcare', 'Healthcare')
dev_sec = dev_sec.mask(dev_sec == 'health', 'Healthcare')
dev_sec = dev_sec.mask(dev_sec == 'Clinical Laboratory', 'Healthcare')
dev_sec = dev_sec.mask(dev_sec == 'banking', 'Finance')
dev_sec = dev_sec.mask(dev_sec == 'financial, credit reporting', 'Finance')
dev_sec = dev_sec.mask(dev_sec == 'financial service company', 'Finance')
dev_sec = dev_sec.mask(dev_sec == 'financial', 'Finance')
dev_sec = dev_sec.mask(dev_sec == 'government', 'Government')
dev_sec = dev_sec.mask(dev_sec == 'political', 'Government')
dev_sec = dev_sec.mask(dev_sec == 'personal and demographic data about residents and their properties of US', 'Government')
dev_sec = dev_sec.mask(dev_sec == 'government, database', 'Government')
dev_sec = dev_sec.mask(dev_sec == 'government, military', 'Government')
dev_sec = dev_sec.mask(dev_sec == 'military', 'Government')
dev_sec = dev_sec.mask(dev_sec == 'retail', 'Retail')
dev_sec = dev_sec.mask(dev_sec == 'Consumer Goods', 'Retail')

info_sec = info['sector'].mask(info['sector'] == 'government', 'Government')
info_sec = info_sec.mask(info_sec == 'military', 'Government')
info_sec = info_sec.mask(info_sec == 'government, military', 'Government')
info_sec = info_sec.mask(info_sec == 'retail', 'Retail')
info_sec = info_sec.mask(info_sec == 'finance', 'Finance')
info_sec = info_sec.mask(info_sec == 'health ', 'Healthcare')
info_sec = info_sec.mask(info_sec == 'misc, health', 'Healthcare')
info_sec = info_sec.mask(info_sec == 'web', 'Technology')
info_sec = info_sec.mask(info_sec == 'web ', 'Technology')
info_sec = info_sec.mask(info_sec == 'tech', 'Technology')
info_sec = info_sec.mask(info_sec == 'tech, app', 'Technology')
info_sec = info_sec.mask(info_sec == 'tech, web', 'Technology')
info_sec = info_sec.mask(info_sec == 'web, tech', 'Technology')
info_sec = info_sec.mask(info_sec == 'app', 'Technology')
info_sec = info_sec.mask(info_sec == 'web, gaming', 'Technology')

bar_x = ['Technology', 'Retail', 'Government', 'Healthcare', 'Finance']
bar_huz = [huz[huz['Industry'] == 'Technology']['Industry'].count(),
           huz[huz['Industry'] == 'Retail']['Industry'].count(),
           huz[huz['Industry'] == 'Government']['Industry'].count(),
           huz[huz['Industry'] == 'Healthcare']['Industry'].count(),
           huz[huz['Industry'] == 'Finance']['Industry'].count()]
bar_dev = [dev_sec[dev_sec == 'Technology'].count(),
           dev_sec[dev_sec == 'Retail'].count(),
           dev_sec[dev_sec == 'Government'].count(),
           dev_sec[dev_sec == 'Healthcare'].count(),
           dev_sec[dev_sec == 'Finance'].count()]
bar_info = [info_sec[info_sec == 'Technology'].count(),
            info_sec[info_sec == 'Retail'].count(),
            info_sec[info_sec == 'Government'].count(),
            info_sec[info_sec == 'Healthcare'].count(),
            info_sec[info_sec == 'Finance'].count()]

to_write += '!!! Sector comparisons\n'
to_write = to_write + ', '.join(bar_x) + '\n'
to_write += 'Huz dataset\n'
to_write = to_write + str(bar_huz) + '\n\n'
to_write += 'Devastator dataset\n'
to_write = to_write + str(bar_dev) + '\n\n'
to_write += 'Info dataset\n'
to_write = to_write + str(bar_info) + '\n\n'

x = pd.Series(range(5))
width = 0.2
plt.bar(x-0.2, bar_huz, width, color='blue', alpha=0.5, log=True)
plt.bar(x, bar_dev, width, color='red', alpha=0.5, log=True)
plt.bar(x+0.2, bar_info, width, color='green', alpha=0.5, log=True)
plt.legend(['Huz', 'Devastator', 'Info'])
plt.xticks(x, bar_x)
plt.show()

# Compare distributions/summaries of dates
to_write += '!!! Date comparisons\n'
to_write += 'Huz dataset\n'
to_write = to_write + huz['Year'].describe().to_string() + '\n\n'
to_write += 'Devastator dataset\n'
to_write = to_write + dev['Year'].describe().to_string() + '\n\n'
to_write += 'Info dataset\n'
to_write = to_write + info['year'].describe().to_string() + '\n\n'

plt.hist(huz['Year'],
         alpha=0.5,
         label='Huz',
         color='blue',
         density=True)
sns.kdeplot(huz['Year'], color='blue', bw_method=0.6)
plt.hist(dev['Year'],
         alpha=0.5,
         label='Dev',
         color='red',
         density=True)
sns.kdeplot(dev['Year'], color='red', bw_method=0.6)
plt.hist(info['year'],
         alpha=0.5,
         label='Info',
         color='green',
         density=True)
sns.kdeplot(info['year'], color='green', bw_method=0.6)
plt.legend(loc='upper right')
plt.title('Year distribution')
plt.show()

# Compare duplicate events between dev and info
comp_dev = dev.drop(columns=['ID', 'Organization type', 'Method', 'Sources'])
comp_info = info.drop(columns=['alternative name', 'date', 'story', 'sector', 'method', 'interesting story',
         'data sensitivity', 'displayed records', 'source name', '1st source link',
         '2nd source link', 'ID', 'Unnamed: 11'])
comp_info = comp_info.rename(columns={
         'organisation': 'Entity', 'year': 'Year', 'records lost': 'Records'
})
temp = comp_info['Year']
comp_info['Year'] = comp_info['Records']
comp_info['Records'] = temp
comp_info = comp_info.rename(columns={
          'Year': 'Records', 'Records': 'Year'
})
dupes = comp_info.merge(comp_dev, on=['Entity', 'Year', 'Records'], how='inner', indicator=True)

to_write += '!!! Devastator and Info comparisons\n'
to_write += 'Event is identified by company name, year, and number of records breached\n'
to_write += 'Number of events that are both in Devastator and Info; ' + str(dupes.shape[0]) + '\n'
to_write += 'Number of events in Devastator but not in Info; ' + str(dev.shape[0] - dupes.shape[0]) + '\n'
to_write += 'Number of events in Info but not in Devastator; ' + str(info.shape[0] - dupes.shape[0]) + '\n'

f = open('verfication.txt', 'w')
f.write(to_write)
f.close()
