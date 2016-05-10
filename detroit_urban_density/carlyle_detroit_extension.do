* Carlyle Davis
* 
* Extension of existing group project for Urban Systems
* Examines detroit's urban density population as compared to cities with
* populations of 1-2.5 million residents (2010 Census)


clear all

* import data
* data at link above is the reelvant subset from
* https://www.census.gov/population/metro/data/pop_pro.html
* cities with pops 1-2.5 million, density as a distance from city center\

use "https://www.dropbox.com/s/s3ql5h3qli1c02y/dataframe_detroit.dta?dl=1", clear

*get list of variable names
. ds

* graph detroit's density vs. average density for cities w/ pop of 1-2.5 mil

graph twoway line Average_of_U_~1 Detroit_Wa~o miles_from_city_center


*graph Detroit vs. other cities with similiar distributions

graph twoway line Detroit_Wa~o Sacramento~i Salt_Lake_~a Orlando_Ki~e Las_Vegas_~e Buffalo_Ni~_ Cleveland_~r miles_from_city_center

