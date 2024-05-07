#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np



import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


# In[3]:


data = pd.read_csv("C:/Users/aravi/Downloads/covid_worldwide.csv",header =0)
data.head()


# In[ ]:





# In[5]:


# Data cleaning
data = data.drop(['Serial Number'],axis =1)


# In[6]:


data.dtypes


# In[7]:


#data cleaning
#our data is having , which makes the data type as string object. so, converting into float
columns = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Active Cases', 'Total Test', 'Population'] 

data[columns] = data[columns].apply(lambda x: x.str.replace(',', '').astype('float'))


# In[8]:


data.dtypes


# In[9]:


#Renaming columns
data =data.rename(columns={"Total Cases":"Cases","Total Deaths":"Deaths","Total Recovered":"Recovered","Active Cases":"Active_Cases","Total Test":"Tests"})


# In[10]:


data.head()


# In[11]:


#HAndling missing values
data.isnull().any()


# In[12]:


data.isnull().sum()


# In[13]:


#recoverd column has more null values, 
# Checking total_recovered missing values

data[data['Recovered'].isna()]


# In[14]:


data.dropna(subset=['Recovered'],inplace=True)
data[data['Recovered'].isna()]


# In[15]:


# population is an important factor for a country and it cannot be null
data[data['Population'].isna()]


# In[16]:


#China is a biggest country but according to data considering it  as Nan is not acceptable.
#so, gere i am adding  the china population value from internet accroding to 2020 census.
data.loc[90, 'Population'] = 1455033020


# In[17]:


data.loc[90]


# In[18]:


#removing the other population null countries mentioned in rthe dataset, where they are not exactly countries
# without population it cannot be called as a country
data.drop([226,229], inplace =True)
data[data['Population'].isna()]


# In[19]:


# replacing total deaths with 0 as mostly null values refers similar
data['Deaths'].fillna(0, inplace=True)


# In[24]:


#data =data.drop(['population'],axis =1)


# In[20]:


data.head()


# In[26]:


# Data Exploration
len(data)


# In[21]:


top5Active= data.sort_values('Active_Cases', ascending=False).head(5)[['Country','Active_Cases']]


# In[72]:


data['% Cases'] = (data['Cases'] / data['Population'])*100
data['% Deaths'] = (data['Deaths'] / data['Population'])*100
data['% active'] = (data['Active_Cases']/data['Cases'])*100

data


# In[23]:


#1. Top 10 countris with most and less deaths
top_deaths= data[['Country','Deaths']].sort_values(by="Deaths",ascending=False)[:10]
top_deaths

less_deaths = data[['Country','Deaths']].sort_values(by="Deaths",)[1:11]
less_deaths


# In[24]:


#1. Top 10 countris with most and less deaths
plt.figure(figsize=(14,15))
plt.subplot(2,2,1)
plt.bar(top_deaths['Country'],top_deaths['Deaths'],color='maroon',label='deaths',width=0.5)
plt.legend()
plt.xlabel("Country")
plt.ylabel("No of death")
plt.title("Top 10 countries with most number of deaths")
plt.xticks(rotation=90)

plt.subplot(2,2,2)
plt.bar(less_deaths['Country'],less_deaths['Deaths'],color='blue',label='deaths',width=0.5)
plt.legend()
plt.xlabel("Country")
plt.ylabel("No of death")
plt.title("Top 10 countries with least number of deaths")
plt.xticks(rotation=90)



# In[25]:


# which country seen the highest percentage of deaths according to its population
#which country affected mostly

data['% Cases'] = (data['Cases'] / data['Population'])*100
data['% Deaths'] = (data['Deaths'] / data['Population'])*100

data


# In[54]:


data['% active']= (data['Active_Cases']/data['Cases'])*100
data


# In[70]:


data['%recover'] = (data['Recovered']/data['Cases'])*100
data['%tests'] = (data['Tests']/data['Population'])*100
data


# In[28]:


import plotly.express as px

per_death = pd.DataFrame(data.groupby('Country')[['Country','% Deaths']].mean().sort_values('% Deaths', ascending=False).round(2).head(10))
fig = px.bar(per_death, x = per_death.index, y = '% Deaths',
            title = ' Highest Death percentage according to Country Population ', template = 'seaborn', color = per_death.index, text = '% Deaths')
fig.show()
per_death


# In[71]:


per_case = pd.DataFrame(data.groupby('Country')[['Country','%recover']].mean().sort_values('%recover',ascending= False).round(2).head(10))
fig = px.bar(per_case, x = per_case.index, y = '%recover',
            title = ' Highestrecovery percentage Countries ', template = 'seaborn', color = per_case.index, text = '%recover')
fig.show()
per_case


# In[74]:


# Countries with Active cases

country_case = data[['Country', 'Active_Cases']]

country_case_plot = country_case.sort_values('Active_Cases', ascending=False).head(10)

sns.set_style("darkgrid")

ccplot = sns.barplot(x='Active_Cases', y='Country', data=country_case_plot)
ccplot.set_ylabel('Country')
ccplot.set_xlabel('Active Cases')
ccplot.set_title('Top Countries with Total Active Cases')


print(country_case_plot)


# In[81]:


#describes how deaths dependent on cases. its a regression model and it plots the data and draws a centre line which is the estimates.

reg_data= data[['Cases','Recovered']]
sns.regplot(data=reg_data, x='Cases', y='Recovered')
plt.xlabel('Total Cases')
plt.ylabel('Total Recoveries')
plt.title('Total Cases vs Total recoveries')
plt.show()


# In[34]:


# is the population factor what makes the country covid cases more seroius ?

plt.subplot(2,1,1)
sc_case = sns.scatterplot(data=data, x='Population', y='Deaths')

sc_case.set_xscale("log")
sc_case.set_yscale("log")

sc_case.set_xlabel('Population', fontweight='bold')
sc_case.set_ylabel('Deaths', fontweight='bold')
sc_case.set_title("Correlation Between Population and deaths", fontweight='heavy')

plt.show()

#plt.xticks(rotation=90)

plt.subplot(2,1,2)
sc_case = sns.scatterplot(data=data, x='Population', y='Cases')

sc_case.set_xscale("log")
sc_case.set_yscale("log")

sc_case.set_xlabel('Population', fontweight='bold')
sc_case.set_ylabel('Total cases', fontweight='bold')
sc_case.set_title("Correlation Between Population and Total Cases", fontweight='heavy')

plt.show()
#plt.xticks(rotation=90)



# In[35]:


import seaborn as sns
sns.set_theme()

# Load the penguins dataset
#penguins = sns.load_dataset("data")

# Plot sepal width as a function of sepal_length across days
g = sns.lmplot(
    data= data,
    x="Cases", y="Recovered"
)

# Use more informative axis labels than are provided by default
g.set_axis_labels( " Total cases","Recovered")


# In[36]:


data.head()


# In[37]:




#cnt = data[['Country','Cases']]
cntry = data.sort_values("Cases", ascending = False).head(10)
cntry


# In[49]:


# Example data


""""species = cntry[["Country"]]
sex_counts = {
    'Active_Cases': np.array([1741147.0,1755.0,95532.0,216022.0,208134.0,10952618.0,422703.0,251970.0,50102.0,207580.0]),
    'Recovered': np.array([101322779.0,44150289.0,39264546.0,37398100.0,35919372.0,21567425.0,29740877.0,25014986.0,24020088.0,21356008.0])
    
}


width = 0.6  # the width of the bars: can also be len(x) sequence


fig, ax = plt.subplots()
bottom = np.zeros(3)

for sex, sex_count in sex_counts.items():
    p = ax.bar(species, sex_count, width, label=sex, bottom=bottom)
    bottom += sex_count

    ax.bar_label(p, label_type='center')

ax.set_title('Number of penguins by sex')
ax.legend()

plt.show()"""


# In[ ]:



conda install geopandas





# In[75]:


#Importing World Map File
import geopandas as gpd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world


# In[76]:


wd = data.merge(world, left_on='Country', right_on='name')

wdf = gpd.GeoDataFrame(wd)




# In[57]:


# which countries economy factor got damaged by covid
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('Covid deaths by Country')
wdf.plot(column='% Deaths', legend=True, cmap='inferno', ax=ax)
plt.show()


# In[77]:


#Countries most cases had

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('Covid Active Cases by Country')
wdf.plot(column='% active', legend=True, cmap='OrRd',ax=ax)
plt.show()


# In[69]:


#Countries most cases had

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title('Covid Cases by Country')
wdf.plot(column='% Cases', legend=True, cmap='OrRd',ax=ax)
plt.show()



top_recover =data[['Country','Recovered','Cases']].sort_values('Recovered', ascending=False).head(5)
top_recover


# In[80]:


# Top 10 Countries with the most Covid Cases , how the people current status of  their stats ////and The amount of people who recovered according to their population.

case_recov = data[['Country', 'Active_Cases','Recovered','Deaths']]

case_recov_df = case_recov.head(5)

case_recov_df.sort_values('Deaths',ascending = False).plot(x='Country', kind='bar', stacked=False)

plt.title("Top 10 Countries with the Most covid cases and stats of recovered, deaths, Active_cases", weight='heavy')
plt.xlabel("Cases (in hundred millions)", weight='bold')
plt.ylabel("Country", weight='bold')

plt.legend(['Active_Cases','Recovered','Deaths'])

plt.show()





# Top 10 Countries with the most Covid Cases and The amount of people who recovered according to their population.

case_test = data[['Country', 'Population','Tests','Cases']]

case_test_df = case_test.head(5)

case_test_df.sort_values('Cases',ascending = False).plot(x='Country', kind='bar', stacked=False)

plt.title("Top 10 Countries with the Most cases and test done in their country", weight='heavy')
plt.xlabel("Cases, ppulation, tests, (in hundred millions)", weight='bold')
plt.ylabel("Country", weight='bold')

plt.legend(['Population','Tests','Cases'])

plt.show()




#"Top 10 Countries with Most covid cases and their population"

case_pop = data[['Country', 'Population','Cases']]

case_pop_df = case_pop.head(10)

case_pop_df.sort_values('Cases',ascending =False).plot(x='Country', kind='bar')

plt.title("Top 10 Countries with Most covid cases and their population", weight='heavy', fontsize='x-large')
plt.xlabel("Cases (in hundred millions)", weight='bold')
plt.ylabel("Country", weight='bold')

plt.legend(['Population','Total Cases'])

plt.show()

print(case_pop_df)







