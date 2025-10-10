# Thesis

## Work performed
- Scraped mental health data from DEIS
- Data is indexed by county and year
- Data includes multiple categories (visits, interventions, etc.)
- Created a map with the scraped data (mapa-salud-mental.cl)
- did simple analisys of data to show the usefulness

## Methodology
### Data Acquisition
The data is available in a SASVisualAnalytics website. The website publishes a variety of reports about mental health in Chile.

Examples of reports:
- Admissions by month
- Attendances by sex
- Egress by age
- Substance abuse by substance
- etc.

Each report can be queried by a variety of filters.

To download the data, a scraper was developed in Python. The scraper downloads the reports indexed by health establishment. The reports themselves are internally indexed by year.

The scraper uses Python Requests to query the website directly, emulating the webUI to query for the desired reports, the response is then stored in json format.

To query for the reports, payloads were created by analyzing the network traffic to the website while querying for reports manually. The payloads are then sent to the website modified to query for the desired establishment.

In total, 125 268 reports in JSON format were downloaded.

### Data Processing
The reports were ingested into an SQL database. The following tables were created

- commune
- establishment
- report (type)
- data

The data was deduplicated, normalized and ingested into the database.

Additionally, a single CSV with all data was created for ease of use.

### Data visualization
The data was used to create a map of mental health in Chile. The map is available at [mapa-salud-mental.cl](https://mapa-salud-mental.cl)

The map can be used to visualize the data per report and per year, showing a heatmap of the selected region. The map AND the data can aditionally be downloaded.

### Data analysis
Simple data analysis was performed over the scraped dataset, gleaming some interesting insights.

All analysis was performed over the "Admissions per month" report, as the monthly reports have higher time granularity and the admissions report encompasses most other reports

#### Global Admissions
![admissions](./screenshots/global.png)

The above graph shows the global number of admissions. There is a clear trend of increase in admissions over the years.

In 2020 there was a drastic decrease in admissions, which can be attributed to the COVID-19 pandemic and the lockdown measures that were implemented.

The purple line is a polynomial regression of degree 2 while the red line is a linear regression. While the polynomial regression fits the data better, it overestimates growth due to the decrease in admissions in 2020. The linear regression may be more accurate for future predictions.

#### Admissions in COSAM NUNOA
![admissions in cosam nunoa](./screenshots/cosam_nunoa.png)

The above graph shows the number of admissions in the COSAM NUNOA establishment, located in the commune of Ñuñoa. There is a clear trend of increase in admissions over the years, with no significant decrease due to COVID-19.

The red line is a linear regression, showing a clear increase.

#### Establishments with most and least growth
Linear regressions were trained for every establishment in Chile. The following table shows the top 5 establishments with the highest and lowest growth rates, extracted by the slope of the linear regression.

| Establishment | Growth Rate |
| --- | --- |
| HOSPITAL CLINICO REGIONAL                                   | 0.132442  |
| CENTRO DE REFERENCIA DE SALUD HOSPITAL PROVINCIA CORDILLERA | 0.074256  |
| HOSPITAL DR CESAR GARAVAGNO BUROTTO                         | 0.064746  |
| CENTRO DE SALUD FAMILIAR DR ADALBERTO STEEGER               | 0.060012  |
| CENTRO DE SALUD FAMILIAR LA FLORIDA                         | 0.056093  |
| ...                                                           | ...         |
| CENTRO DE SALUD FAMILIAR DR EDELBERTO ELGUETA               | -0.044212 |
| COMPLEJO HOSPITALARIO DR SOTERO DEL RIO                     | -0.059438 |
| CENTRO DE SALUD FAMILIAR DE MOLINA                          | -0.099697 |
| COSAM LA GRANJA                                             | -0.197678 |
| CENTRO COMUNITARIO DE SALUD FAMILIAR BEATO PADRE HURTADO    | -0.413375 |

The following scatter plots show the data for the top 2 and worst 2 establishments.

The high score of this establishment can be attributed to the low number of patients with 2024 being an outlier with an unusually large amount of patients. This may be due to a change in how the data is collected or reported.
![Hospital Clinico Regional](screenshots/hospital_clinico_regional.png)

This establishments shows an increasing trend and may be better fit by a  polynomial regression.
![Centro de referencia hospital provincia cordillera](screenshots/provincia_cordillera.png)

This establishment shows 2 distinct trends, one from 2019 to 2021 which shows a constant number of patients. And a second one from 2022 onwards which shows an increasing linear trend.
![HOSPITAL DR CESAR GARAVAGNO BUROTTO](screenshots/cesar.png)

This establishment shows a drastic decline in 2020, with no recovery. This may be due to the COVID-19 pandemic, but we suspect that the data is mixed with a second report, or the way the data is reported changed.
![COSAM LA GRANJA](screenshots/granja.png)

This establishment low score can be attributed to the low amount of data, starting only in 2024, and the unusually high number of patients for april.
![screenshots/beato.png](screenshots/beato.png)