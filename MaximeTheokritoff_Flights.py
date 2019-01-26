#MaximeTheokritoff_Assignment3.py

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


"""
    In this program you will be analyzing airline flight performance data to answer several questions. You should go to
    https://www.transtats.bts.gov/Tables.asp?DB_ID=120&DB_Name=Airline%20On-Time%20Performance%20Data
    to download a csv file containing your data. You should choose a month to analyze that has a major holiday in it.
    On the download page you should select the following fields of information for your database:
    FlightDate, Carrier, Origin, OriginState, Dest, DestState, CRSDepTime, DepTime, TaxiOut, TaxiIn,
    CRSArrTime, ArrTime, Cancelled, Diverted, CRSElapsedTime, ActualElapsedTime, AirTime, and Distance.
    Download the selected data file. It will be quite large.
"""

"""
    You should read in the file data. First remove all flights that were canceled or diverted from the set to analyze.
"""
df = pd.read_csv("FlightDataDec2017.csv", sep=',',header=0)
df = df[df.CANCELLED != 1 ]
df = df[df.DIVERTED != 1 ]

"""
    padding1(x) reformat the time data for it to be analysed 800 -> 0800 (08h 00 min)
    padding2(y) reformat the time data for it to be analysed 852.0 -> 0852 (08h 52 min)
    timediff(x,y) reformat the time data for it to be analysed leveraging padded1 and padded2:
        it takes the hours of both and substratcs tem then multiplies it by 60 so we get the result in minuts.
        it then takes the minuts and substracts them
        finaly it adds the two results giving us a time difference in minuts.
"""

def timediff(x,y):
    padX = padding1(x)
    padY = padding2(y)
    return (60*(int(padX[:2])-int(padY[:2])) + int(padX[2:])-int(padY[2:]))

def padding1(x):
    paddedX = str(x)
    if len(paddedX)-2 == 1:
        paddedX = "240" + paddedX[:1]
    elif len(paddedX)-2 == 2:
        paddedX = "24"+ paddedX[:2]
    elif len(paddedX)-2 == 3:
        paddedX = "0"+ paddedX[:3]
    else:
        paddedX = paddedX[:4]
    return paddedX

def padding2(y):
    paddedY = str(y)
    if len(paddedY) == 1:
        paddedY = "240" + paddedY
    elif len(paddedY) == 2:
        paddedY = "24"+ paddedY
    elif len(paddedY) == 3:
        paddedY = "0"+ paddedY
    return paddedY




"""
    Then you should use your pandas magic to determine the following:
    What are the top 5 airlines in terms of:
    ontime departure percentage?
    ontime arrival percentage?
    average departure lateness?
    average arrival lateness?
"""


"""
    Creating the new columns to know if a flight is late.
    Negative values correspond to flights leaving or arriving early
"""

"""
    For DEP_LATE and ARR_LATE, a negative value means the flight is on time, a positive value means it isn't.
"""


df = df.assign(DEP_LATE = np.vectorize(timediff)(df["DEP_TIME"], df["CRS_DEP_TIME"]))
df = df.assign(ARR_LATE = np.vectorize(timediff)(df["ARR_TIME"], df["CRS_ARR_TIME"]))

dfCarLateDep = df[df.DEP_LATE > 0].groupby("CARRIER")["DEP_LATE"].mean().sort_values(ascending=False)[:5]
dfCarLateArr = df[df.ARR_LATE > 0].groupby("CARRIER")["ARR_LATE"].mean().sort_values(ascending=False)[:5]

print("TOP 5 carrier Late departure")
print(dfCarLateDep)
print()
print("TOP 5 carrier Late arrival")
print(dfCarLateArr)
print()

def isOnTime(x):
    if x<0:
        return 1
    else :
        return 0

"""
    For ON_TIME_DEP and ON_TIME_ARR, 1 means the flight is on time, 0 means it isn't.
"""

df = df.assign(ON_TIME_DEP = df["DEP_LATE"].apply(isOnTime))
df = df.assign(ON_TIME_ARR  = df["ARR_LATE"].apply(isOnTime))

onTimeDep = df.groupby(["CARRIER", "ON_TIME_DEP"]).size().unstack()
onTimeArr = df.groupby(["CARRIER", "ON_TIME_ARR"]).size().unstack()

onTimeDep = onTimeDep.assign(PERCENTAGE_ON_TIME_DEP = (onTimeDep[1]/(onTimeDep[1]+onTimeDep[0])))
onTimeArr = onTimeArr.assign(PERCENTAGE_ON_TIME_ARR = (onTimeArr[1]/(onTimeArr[1]+onTimeArr[0])))


print("Top 5 based on percentage of flights departing on time per Carrier")
print(onTimeDep["PERCENTAGE_ON_TIME_DEP"].sort_values(ascending=False)[:5])
print()
print("Top 5 based on percentage of flights arriving on time per Carrier")
print(onTimeArr["PERCENTAGE_ON_TIME_ARR"].sort_values(ascending=False)[:5])
print()

"""
    What are the relative percentages of flights that leave and arrive on time vs.
    leave on time and arrive late vs. leave late and arrive on time, and leave and arrive late?
"""

"""
    We will assigne 4 values:
        1 Leaves and arrives on time
        2 leaves on time and arrives late
        3 leaves late and arrives on time
        4 leaves and arrives late
"""

def LeaveArrive(x,y):
    if x == 1 and y == 1:
        return 1
    elif x == 1 and y == 0:
        return 2
    elif x == 0 and y == 1:
        return 3
    else :
        return 4

df = df.assign(LEAVES_ARRIVE = np.vectorize(LeaveArrive)(df["ON_TIME_DEP"], df["ON_TIME_ARR"]))

leavesArrives = df.groupby(["LEAVES_ARRIVE"]).aggregate('size')

sumLeavesArrives = leavesArrives.sum()

print("Distribution of flights based on departure and arrival times.")
print("{0:.2f}%".format(leavesArrives[1]/sumLeavesArrives*100), " of flights leaft and arrived on time")
print("{0:.2f}%".format(leavesArrives[2]/sumLeavesArrives*100), " of flights leaft on time but arrived late")
print("{0:.2f}%".format(leavesArrives[3]/sumLeavesArrives*100), " of flights leaft late but arrived on time")
print("{0:.2f}%".format(leavesArrives[4]/sumLeavesArrives*100), " of flights leaft and arrived late")
print()

"""
    Do states in the northeast (Pennsylvania, Maryland, New Jersey, Delaware, New York, Connecticut, Rhode Island,
    Massachusetts, Vermont, New Hampshire, and Maine) have worse time performance than the rest of the country
    based on average departure lateness?
"""

#northWestStates = ["PA", "MD", "NJ", "DE", "NY", "CT", "RI","MA", "VT", "NH",  "ME"]
stateLateness = df.groupby("ORIGIN_STATE_ABR")["DEP_LATE"].mean()

NorthEastLateness = stateLateness[["PA", "MD", "NJ", "DE", "NY", "CT", "RI","MA", "VT", "NH",  "ME"]].mean()
RestOfUSLatness = (stateLateness.sum() - stateLateness[["PA", "MD", "NJ", "DE", "NY", "CT", "RI","MA", "VT", "NH",  "ME"]].sum())/( stateLateness.count()- stateLateness[["PA", "MD", "NJ", "DE", "NY", "CT", "RI","MA", "VT", "NH",  "ME"]].count())

print("Average Lateness of North Eastern flights")
print(NorthEastLateness)
print("Average Lateness of the rest of the US flights")
print(RestOfUSLatness)
print()


"""
    What are the top 5 airports in terms of:
    Ontime departure percentage?
    Ontime arrival percentage?
    Average departure lateness?
    Average arrival lateness?
"""

dfAirportOnTimeDep = df[df.DEP_LATE < 0].groupby("ORIGIN")["DEP_LATE"].mean().sort_values()[:5]
dfAirportOnTimeeArr = df[df.ARR_LATE < 0].groupby("DEST")["ARR_LATE"].mean().sort_values()[:5]

print("Top 5 airport for on time departure")
print(dfAirportOnTimeDep.abs())
print("Top 5 airport for on time arrival")
print(dfAirportOnTimeeArr.abs())
print()

AirportonTimeDep = df.groupby(["ORIGIN", "ON_TIME_DEP"]).size().unstack()
AirportonTimeArr = df.groupby(["DEST", "ON_TIME_ARR"]).size().unstack()

AirportonTimeDep = onTimeDep.assign(PERCENTAGE_ON_TIME_DEP = (onTimeDep[1]/(onTimeDep[0]+onTimeDep[1])))
AirportonTimeArr = onTimeArr.assign(PERCENTAGE_ON_TIME_ARR = (onTimeArr[1]/(onTimeArr[0]+onTimeArr[1])))


print("Top 5 based on percentage of flights departing on time per Airport")
print(AirportonTimeDep["PERCENTAGE_ON_TIME_DEP"].sort_values(ascending=False)[:5])
print()
print("Top 5 based on percentage of flights arriving on time per Airport")
print(AirportonTimeArr["PERCENTAGE_ON_TIME_ARR"].sort_values(ascending=False)[:5])
print()

"""
    For each of five major airports - LAX, JFK, DFW, ORD and ATL - what are the top 5
    airlines in terms of the longest taxi out and taxi in times?
"""

majorAirports = ["LAX", "JFK", "DFW", "ORD", "ATL"]

majorAirportsTaxiOut = df.groupby(["ORIGIN","CARRIER"], sort=True)["TAXI_OUT"].mean()
majorAirportsTaxiIn = df.groupby(["DEST","CARRIER"])["TAXI_IN"].mean()


print("Top 5 airlines in terms of the longest taxi out")
for item in majorAirports:
    print(item)
    print( majorAirportsTaxiOut[item].sort_values(ascending=False)[:5])
    print()

print("Top 5 airlines in terms of the longest taxi in")
for item in majorAirports:
    print(item)
    print( majorAirportsTaxiIn[item].sort_values(ascending=False)[:5])
    print()

print("Average Lateness of North Eastern flights")
print(NorthEastLateness)
print("Average Lateness of the rest of the US flights")
print(RestOfUSLatness)
print()

"""
    What are the relative percentages of flight time in terms of taxi out time,
    actual flying time,
    and taxi in time?
"""



print("Relative percentage of flight time in term of taxi in time: ", "{0:.2f}%".format((df["TAXI_IN"].sum()/df["ACTUAL_ELAPSED_TIME"].sum())*100))
print("Relative percentage of flight time in term of taxi out time: ", "{0:.2f}%".format((df["TAXI_OUT"].sum()/df["ACTUAL_ELAPSED_TIME"].sum())*100))
print("Relative percentage of flight time in term of actual flying time: ", "{0:.2f}%".format((df["AIR_TIME"].sum()/df["ACTUAL_ELAPSED_TIME"].sum())*100))
print()

"""
What are the top 5 airlines in terms of actual flight speed?
"""
df["FLIGHT_SPEED"] = df["DISTANCE"]/(df["AIR_TIME"]/60)

print("Top 5 airlines in terms of actual flight speed (given in miles/hour)")
print(df.groupby("CARRIER")["FLIGHT_SPEED"].mean().sort_values(ascending=False)[:5])
print()

"""
    Do flights that depart late fly faster than those that do not?
    (compare flying times of the same routes when they are late and not)
"""

"""
    For ON_TIME_DEP and ON_TIME_ARR, 0 means the flight is on time, 1 means it isn't.
"""

FasterFlight = df.groupby(["ORIGIN", "DEST", "ON_TIME_DEP"])[ "AIR_TIME"].mean().unstack().dropna()

#OnTimeSpeed = pd.Series(FasterFlight["ON_TIME_DEP" == 0]-FasterFlight["ON_TIME_DEP" == 1])

def isFaster (x,y):
    if x>y :
        return 1
    else :
        return 0

"""
    If LATE_IS_FASTER is 0 this means that a late flight is not fater if it is 1 it means a late flight is faster
"""

FasterFlight = FasterFlight.assign(LATE_FASTER = np.vectorize(isFaster)(FasterFlight[0],FasterFlight[1]))

print(FasterFlight["LATE_FASTER"].value_counts().rename({0: "Not faster", 1: "Faster"}) )
print()

"""
    Do longer flights arrive late more often than shorter flights?
    (plot percentage arrival lateness vs flight length)
"""

def isLate(x):
    if x>0:
        return 1
    else :
        return 0
"""
    If a flight is late returns 1 else return 0
"""
df = df.assign(LATE_ARR_TRUE = df["ARR_LATE"].apply(isLate))

longLate = df.groupby("DISTANCE")["LATE_ARR_TRUE"].sum()
longLate = longLate.to_frame()
longLate = longLate.assign(PERCENTAGE_OF_LATE = 100*longLate/longLate.sum())

#print(longLate)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(longLate.index, longLate["PERCENTAGE_OF_LATE"],color ='#FECA30')
ax.set_ylim((0,0.5))

plt.title('Flight Latenes Percentage Based on Distance')
plt.xlabel('Distance Miles')
plt.ylabel('Percentage Arrival Lateness')
plt.show()

print("Shorter Flights arrive late more often than longer Flights")
print()

"""
    Which flights (morning: before noon, afternoon: 12 - 6, evening: 6-12),
    have a greater on time departure percentage?  on time arrival percentage?
"""
def timeofday(x):
    if x < 1200:
        return "Morning"
    elif x < 1800:
        return "Afternoon"
    else :
        return "Evening"

df = df.assign(TIME_OF_DAY = df["DEP_TIME"].apply(timeofday))


timeDayDep = df.groupby(["TIME_OF_DAY", "ON_TIME_DEP"])["ON_TIME_DEP"].aggregate('count').unstack()
timeDayArr = df.groupby(["TIME_OF_DAY", "ON_TIME_ARR"])["ON_TIME_ARR"].aggregate('count').unstack()

print("On time departure precentages based on periode of the day")
print(timeDayDep[1]/(timeDayDep[1]+timeDayDep[0]))
print("On time arrival precentages based on periode of the day")
print(timeDayArr[1]/(timeDayArr[1]+timeDayArr[0]))
print()

"""
    Which day in the month has a worse performance time based on departure lateness percentage -
    the holiday, the day before, or the day after?
"""
df = df.assign(LATE_DEP_TRUE = df["DEP_LATE"].apply(isLate))

Holiday = df.groupby(["FL_DATE", "LATE_DEP_TRUE"] )["LATE_DEP_TRUE"].aggregate('count').unstack()


print("Percentage of flights late departure on the 2017-12-23: ")
print("{0:.2f}%".format(100*Holiday[1]["2017-12-23"]/(Holiday[0]["2017-12-23"]+Holiday[1]["2017-12-23"])))
print("Percentage of flights late departure on the 2017-12-24: ")
print("{0:.2f}%".format(100*Holiday[1]["2017-12-24"]/(Holiday[0]["2017-12-24"]+Holiday[1]["2017-12-24"])))
print("Percentage of flights late departure on the 2017-12-25: ")
print("{0:.2f}%".format(100*Holiday[1]["2017-12-25"]/(Holiday[0]["2017-12-25"]+Holiday[1]["2017-12-25"])))
print()
"""
    Do flights on the weekend have a worse performance than those on weekdays in terms of arrival lateness percentage?
"""

"""
    weekends: 2,3,9,10,16,17,23,24,30,31
"""
weekend = ["2017-12-02","2017-12-03","2017-12-09","2017-12-10","2017-12-16","2017-12-17","2017-12-23","2017-12-24","2017-12-30","2017-12-31"]
def isWeekend(x):
    if x in weekend :
        return "Weekend"
    else :
        return "Weekday"
df = df.assign(IS_WEEKEND = df["FL_DATE"].apply(isWeekend))

weekendLateness = df.groupby(["IS_WEEKEND", "LATE_ARR_TRUE"] )["LATE_ARR_TRUE"].aggregate('count').unstack()

print("Percentage of flights late on Weekdays")
print("{0:.2f}%".format(100*weekendLateness[1]["Weekday"]/(weekendLateness[1]["Weekday"]+weekendLateness[0]["Weekday"])))
print("Percentage of flights late on Weekends")
print("{0:.2f}%".format(100*weekendLateness[1]["Weekend"]/(weekendLateness[1]["Weekend"]+weekendLateness[0]["Weekend"])))
print()


"""
    Think up three other interesting questions and determine the answers.
"""

"""
    What are the descriptive statistics of flight arriving late and flights arriving on time.
"""

print(df.groupby("LATE_ARR_TRUE" )["ARR_LATE"].describe())
print()
#print(df.groupby(["IS_WEEKEND", "TIME_OF_DAY"] )["DEP_LATE"].describe())


"""
    Is Taxi Out average time longer for flights departing late?
"""
print("Comparing Taxi out time for departing flights, late and on time")
print(df.groupby(["LATE_DEP_TRUE"] )["TAXI_OUT"].mean())
print()
"""
    Is Taxi In average time time longer for flights arriving late?
"""
print("Comparing Taxi in time for departing flights, late and on time")
print(df.groupby(["LATE_ARR_TRUE"] )["TAXI_IN"].mean())
print()


"""
    What is the top 10 of flight route?
    (same departure and arrival)
"""
df = df.assign(COUNT = 1)

flightRoutes = df.groupby(["ORIGIN", "DEST"])["COUNT"].count()

print("Top 10 flight routes")
print(flightRoutes.sort_values(ascending=False)[:10])
print()

"""
    Submit your commented source code, as well as the output it generates for your sample data set.
    You don't need to submit the data set, just indicate what month/year you analyzed.
"""

#df.to_csv("done.csv", sep=',')
