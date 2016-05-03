# Urban Systems
**Counting Public Roadspace in Gurgaon, India**

Carlyle Davis


Upon a [recent article in CityLabs](http://bit.ly/1VxNBSD) about the privitization of public services in
the city of Gurgaon, India, I set out to quickly calculate the percentage of land
devoted to "public" roads that companies will have to upkeep.  This is especially
relevant as the city's administration attempts to find a sponsor to build and maintain
a new light rail system featuring "driverless taxi pods" that will serve the city's
population, and alleviate the traffic burden of the 876,000+ inhabitants.

After obtaining road maps of the city from Google Maps, I converted the images into
black/white images, and counted the number of black pixels (roads / highways) vs.
the total pixels in each image to obtain a rough percentage of public space devoted to roads.

**I calculate that ~8% of the land in Gurgaon India is devoted to roads and highways.**

As a comparison, I also calculated the amount of public space in the English planned city
of New Delhi (of which Gurgaon is a 'satellite city').  With planned public space and
a very planned layout, New Delhi's public space allocation is more in line with
other cities of similiar size.  For comparison's sake, I simply took a a small area from
the clearly planned city-center.

**I calculate that ~14% of the land in New Delhi, India is devoted to roads and highways.**

This is an interesting snapshot of the risk that cityplanners take when they do not
set aside public space from the start, and allow free markets to determine how people
share space and move through a city.

One way this project could be improved is to include parks, plazas, and other large
gathering space for the city, in order to have a rough approximate for total public space.

For code and/or to reproduce my results, please see the Ipython Notebook in this
folder, and the included image files.
