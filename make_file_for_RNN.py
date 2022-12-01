import pandas as pd
import math
import matplotlib.pyplot as plt

# import seaborn as sns

data_frame = pd.read_csv('Session-Details-Summary-20190329.csv')

data_frame['Start Date'] = pd.to_datetime(data_frame['Start Date'])
data = pd.read_csv('Station-table-all-columns-20190320.csv')
unique_stations = data.drop_duplicates(subset=['Address 1'])

start_date = pd.to_datetime('01-01-2018')
start=[0,7,11,17,19]
end=[6,10,16,18,23]

lebely = []
# for i in range(365):
#     dat = start_date - pd.to_timedelta(i, 'd')
#     date = str(dat.year) + "-" + str(dat.month) + "-" + str(dat.day)
#     lebely.append(date)

for i in range(0,len(unique_stations)):
    name = unique_stations['Address 1'].iloc[i]
    print(name)

    dataframe = []
    total = [[0 for x in range(5)] for y in range(453)]

    name = unique_stations['Address 1'].iloc[i]
    for j in range(0, len(data)):
        if unique_stations['Address 1'].iloc[i] == data['Address 1'].iloc[j]:
            print(data['Station Name'].iloc[j])
            single_station = data_frame[data_frame['Station Name'] == data['Station Name'].iloc[j]]

            #
            #
            #     #length=len(single_station)-2
            #
            #     #day_count = (single_station['Start Date'].iloc[0] - single_station['Start Date'].iloc[length]).days + 1
            #     #print(length,day_count)
            #
            #     #print(single_station['Start Date'].iloc[0] + pd.to_timedelta(l, 'd'))
            #
            #
            #

            for da in range(0, 453):

                for m in range(0, 5):
                    dat = start_date + pd.to_timedelta(da, 'd')

                    totalenergy = 0
                    totalenergy = single_station[(single_station['Start Date'].dt.hour >= start[m]) & (single_station['Start Date'].dt.hour <= end[m]) &(
                    single_station['Start Date'].dt.year == dat.year) & (
                                                 single_station['Start Date'].dt.month == dat.month) & (
                                                 single_station['Start Date'].dt.day == dat.day)]

                    total[da][m] += totalenergy['Energy (kWh)'].sum()

                    # print(total[m][da])
    for da in range(0, 453):
        for m in range(0, 5):
            d = []
            dat = start_date + pd.to_timedelta(da, 'd')
            d.append(m)
            d.append(dat.dayofweek // 5)
            d.append(math.sin(dat.dayofyear * 2 * 3.1416 / 365))
            d.append(math.cos(dat.dayofyear * 2 * 3.1416 / 365))
            # if m<=3:
            #     d.append(1)
            #     d.append(0)
            #     d.append(0)
            #     d.append(0)
            #     d.append(0)
            #     d.append(0)
            # elif m<=7:
            #
            #         d.append(0)
            #         d.append(1)
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            # elif m <= 11:
            #
            #         d.append(0)
            #         d.append(0)
            #         d.append(1)
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            # elif m <= 15:
            #
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(1)
            #         d.append(0)
            #         d.append(0)
            #
            # elif m <= 19:
            #
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(1)
            #         d.append(0)
            #
            # elif m <= 23:
            #
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(0)
            #         d.append(1)
            if m==0:
                d.append(math.ceil(total[da-1][4]))
                d.append(math.ceil(total[da-1][3]))
            elif m==1:
                d.append(math.ceil(total[da][0]))
                d.append(math.ceil(total[da-1][4]))
            else:
                d.append(math.ceil(total[da][m-1]))
                d.append(math.ceil(total[da][m-2]))

            d.append(math.ceil(total[da][m]))
            d.append(dat)
            dataframe.append(d)

    df = pd.DataFrame(dataframe, columns=['PeriodofTime', 'Weekday', 'Sin', 'Cos', 'Energypvetimeperiod', 'Energypevpevperiod', 'Energy', 'Date'])
    name += ".csv"

    df.to_csv(name, sep=',')






# plt.figure(figsize=(30, 10))
#    ax = sns.heatmap(total, linewidths=0.5, cmap="Reds")
#    ax.set_yticklabels([23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
#    ax.set_xticklabels(lebely)
#    plt.yticks(rotation=70)
#    xmin, xmax = plt.xlim()
#    ymin, ymax = plt.ylim()
#    name += ' year.png'
#    plt.savefig(name)
#    plt.clf()




