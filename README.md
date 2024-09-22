# Spatial Analysis of Taxi GPS Data (2019)

# Overview
---

Global Positioning System (GPS) data has been a valuable source of information in transportation, urban planning, and logistics. In the Philippines, several transport companies and organization utilized GPS in order to optimize their operational policies to improved revenue and resources. In the government, the use of GPS has been pivotal to improve its key services particularly in public transportation. 

# Project Summary
---

This data has been collected by LTFRB through its mobile big data partners in telco. The time coverage is March 2018 and 2019 in a 24 hour interval. The objective of this project is to understand and analyze the behavior of commuters using taxi as a mode of transportation. Additionally, we have to recommend policy that will help improve the experience of commuters.

**A. Contents**

1. **Temporal Coverage**
    - With the aid of python, I analyze the area covered by the data such as logs per taxi, and the trends that corresponds to it such as daily average ridership.
2. **Stay Point Identification**
    - Using python I conducted a spatial grouping. In order to a GPS point belong to a group, the taxi should stay in a significant amount time and didn’t move or exceed a specified distance.
3. **Spatial Clustering**
    - After identifying the staypoints, I conducted a spatial clustering using DBSCAN and Hierarchical technique. From this two machine learning algorithm, I derived two clustering results that allows me to give recommendations.
4. **Recommendation**
    - In this part, I recommend practical strategies in mobility management in terms of shifting the commuter from taxi to public transportation.

**B. Data Structure**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/23267e73-ef82-42e8-b595-a92749e1e786/image.png)

# I. Temporal Coverage
---

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/eed2a1ff-8403-4d11-b9c7-82410d506390/image.png)

                                                                    Fig 1. GPS Logs per User

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/351cdce6-5afa-48eb-83ce-155113beba1c/image.png)

                                                         Fig 2. Coverage and Logs per Day
                                                         
- The daily GPS logs in this data is recorded in an average interval of 2 minutes. So the logs does not translate to an individual ride. We can see in the user logs that the records for each user is not equal since per user the interval is not equal.
- Using Spatial filtering, I reduced the coverage within `metro manila` with allowances for its adjacent provinces such as *bulacan* in the *north*, *rizal* in the *east*, and *cavite* in the *south*. Logs that reach up to *clark*, and down to *laguna* are clipped.
- For this data to be understand, I utilized its date features and grouped them based on its day and hour. In this way, I will overpower the inaccuracy for the 2 minutes interval of the GPS logs.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/d369f89f-3933-4391-8c5a-7d0f1dd864de/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/03b61dce-2610-4ff9-99c5-c634bf6fb53c/image.png)

                                                                Fig 3. Daily Average Ridership

- **Daily Average Rides:**
    - The daily average rides follow a trend, it has 3 on-peaks: 9 AM, 1 PM, 7 PM
    - 3 off-peaks: 11 AM, 5 PM, 11 PM
    - For 9 AM, Monday recorded the highest ridership, with a 701 average rides.
    - For 1 PM, Tuesday the is highest, with 618 average rides.
    - For 7 PM, Wednesday recorded a 662 average rides.
    - For a month basis, we will divide this ridership in 4 days:
        - 175 cars for every monday morning
        - 155 cars for every tuesday afternoon
        - 166 cars for every wednesday evening
    - Assuming every ride has 1 passenger/commuter, it will require a 175 car to transport 1 person from its origin to destination.
    - If we compressed this by 4 passengers if ride sharing is implemented, it will require approximately 43 cars in monday morning
    - If we compressed this by 60 passengers, if they used a bus or shuttle service, it will require 3 buses to transport the 175 passengers.

# II. Stay Point Identification
---

A stay point is a location identified from multiple GPS logs based on specific criteria. The GPS logs within this location are averaged to determine its latitude and longitude.

**A. Criteria**

1. Define a radius for the basis of the stay point.
2. Feed the GPS data into a loop, and test each point against the following conditions:
    - `time_stayed` ≥ `minimum_time_to_stay`
    - `distance_changed` ≤ `threshold_distance`
3. Starting from the initial GPS point, measure the distance to subsequent GPS logs. If the distance exceeds the `threshold_distance`, exit the loop.
4. Compute the stay point’s latitude and longitude by averaging the GPS logs that meet the criteria.
5. Calculate the `cumulative time` and `cumulative distance` for the stay point.
6. Proceed to the next point to feed in the loop.

**B. Pseudo-code**

```python
class StayPointIdentification:
    def __init__(self, data, cutoff_distance, minimum_time):
        # Initialize variables
        self.gps_data = data
        self.cutoff_distance = cutoff_distance
        self.minimum_time = minimum_time
        self.staypoints = self.identify_staypoints()
        self.staypoints_df = self.to_dataframe()

    def centroid(self, latitude, longitude):
        # Compute centroid of given latitude and longitude lists
        if len(latitude) == 1:
            return sum(latitude) / len(latitude), sum(longitude) / len(longitude)
        return sum(latitude) / (len(latitude) - 1), sum(longitude) / (len(longitude) - 1)

    def radian(self, point):
        # Convert point to radians
        return float(point) * math.pi / 180.0

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Calculate Haversine distance between two GPS points
        radius = 6371  # Earth radius in km
        phi1, phi2 = self.radian(lat1), self.radian(lat2)
        delta_phi = phi2 - phi1
        delta_lambda = self.radian(lon2) - self.radian(lon1)
        
        # Haversine formula
        a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
        c = 2 * asin(sqrt(a))
        return radius * c

    def identify_staypoints(self):
        # Identify staypoints based on cutoff distance and minimum time
        staypoints = []
        for each point in gps_data:
            if distance <= cutoff_distance and time_interval >= minimum_time:
                staypoints.append(compute_centroid())
        return staypoints

    def to_dataframe(self):
        # Convert staypoints to a dataframe, filtering by minimum time
        dataframe = convert_to_dataframe(self.staypoints)
        return filter_dataframe_by_time(dataframe, self.minimum_time)

```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/767b2afb-94d8-4b7f-8eee-c6caa0911058/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/024f4dcc-e845-4669-94bf-19f2b695907b/image.png)

# III. Spatial Clustering

---

1. **Density-Based Spatial Clustering of Application with Noise**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/ebed51a4-8d16-499e-882e-b2a271b78e7b/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/74370dc5-0fa5-41d4-bd20-dc37edcd313f/image.png)

b. **Hierarchical Clustering**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/907fdfaa-fbf1-4da8-b1f3-9e71cb93e094/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/eb80c43b-dd65-485a-a309-32a9150c4197/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3ef1cfee-86df-4a4c-a9c5-d0e7329840df/f1396d9b-497a-478b-9485-d6a0dc0f2491/image.png)

1. High Cumulative Count, High Cumulative Time: This combination may indicate areas of high traffic density and prolonged dwell time, where there is a lot of activity happening. These areas may be urban centers, shopping districts, or entertainment venues.

1. Low Cumulative Count, Low Cumulative Time: This combination may indicate areas of low traffic activity, where there is little movement or activity happening. These areas may be remote or less populated regions.

1. Cumulative count is low and the cumulative time varies: it suggest that the area is not heavily trafficked but that there are some events or activities that draw people to the area for varying amounts of time

# IV. Recommendations
---

1. `High-Demand Areas`: Areas with high demand for taxis can be targeted for investment in public transit system such as point-to-point bus system. The stay points within `Antipolo`, `Taguig`, and `QC` can be redesign to have this bus system and connect them to the main transport network such as the MRT and EDSA Busway. In this way, commuters will be encourage to use public transport that offers minimal the transportation cost and time in changing modes.

1. `Low-Demand Areas`: Areas with low demand for taxi is accompanied of short travel time. This areas indicates that the origin-to-destination distance is short and can be done using other mode of transport such as `cycling` or `walking`. This area can be targeted for green spaces infrastructures, such as  exclusive pedestrian and bicycle lanes. This promotes commuters to change in active transport instead of taxi.

1. `Varying Demand Areas`: This area is accompanied with varying demand and travel time, stay points that falls in this area are present on both residential and business areas. This might indicates that the demand is based only in specific situation of the commuter. For low travel time, approach for `low-demand area` can be adapted. For areas with high travel time, `carpooling` or `ride-sharing` can be implement within the area. This will reduce space in the road, saves cost for users.

1. `Average-Demand Areas`: The stay points that falls in this category are present in business areas, malls, schools, and local communities. The efforts in this category should focus in green space infrastructure such as inclusive waiting area for children, senior citizens, and PWDs.
