select * from googleapps limit 5;


-- What are the top 10 Categories that are installed from the Google Play Store?

SELECT Category, SUM(Installs) AS Total_Installs
FROM googleapps
GROUP BY Category
ORDER BY Total_Installs DESC
LIMIT 10;

-- which is the highest rated categories?

select  Category, avg(Rating) as Avg_Rating
FROM googleapps
GROUP BY Category
ORDER BY Avg_Rating DESC;

-- What are the top 5 paid Apps based with highest ratings and installs?
 
select app_name, max(rating) as max_rating, max(maximum_installs) as max_installs 
from googleapps
where price > 0
group by app_name 
order by max_rating desc, max_installs desc
limit 10;

-- What are the 10 Categories in playstore as per the count?
SELECT Category, COUNT(*) AS App_Count
FROM googleapps
GROUP BY Category
ORDER BY App_Count DESC
LIMIT 10;

--What is highest 5 rated category?
SELECT Category, AVG(Rating) AS Avg_Rating
FROM googleapps
GROUP BY Category
ORDER BY Avg_Rating DESC
LIMIT 5;

-- Count the number of apps in each category
select category, count(*) num_apps
from googleapps 
group by category;


-- Find the top-rated apps
select app_name, rating
from googleapps
order by rating desc
limit 10;

-- Determine the most expensive apps
select app_name, price, currency
from googleapps
where price > 0 
order by price desc;

-- Find the oldest and newest apps
select app_name, released, last_updated
from googleapps 
order by released asc
limit 1;

select app_name, released, last_updated
from googleapps 
order by released desc
limit 1;

-- Identify categories where the majority of apps are ad-supported
select category 
from (select category, sum(case when ad_supported = 'TRUE' then 1 else 0 end ) as ad_support_count, count(*) as total_count
	 from googleapps 
	 group by category
	 ) as subquery
where ad_support_count> total_count/2;


-- Find the apps that have not been updated in the last six months
select app_name, last_updated
from googleapps 
where last_updated < NOW()- interval '6 month';