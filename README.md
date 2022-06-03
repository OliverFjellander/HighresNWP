
<div id="top"></div>
<div align="center">
<h2 align="center">The potential for using sub-kilometer scale numerical weather models for short-term predictions of extreme rainfall</h3>

  <p align="center">
   <b> Technical University of Denmark, DTU Sustain </b> <br>
    Department of Environmental and Resource Engineering <br>
   <em> June 3rd, 2022 </em>
  <br /> <br />
  <em> Produced by <a href="https://github.com/OliverFjellander"><strong>Oliver Hede Fjellander</strong> (s152649) </a>  </em>
  <br />
    <em> Supervised by <strong>Jonas Wied Pedersen</strong></a> and <strong>Peter Steen Mikkelsen</strong></a>  </em>
    
  </p>
</div>

## Preface

This thesis was written by Oliver Hede Fjellander as a fulfillment of the M.Sc. degree in Environmental Engineering at the Technical University of Denmark (DTU). The work consisted of 30 ECTS points conducted from the 3th of January, 2022, until the 3rd of June, 2022. The project was supervised by industrial postdoc Jonas Weid Pedersen at Danish Meteorological Institute (DMI) and DTU environment together with Professor Peter Steen Mikkelsen from DTU environment. Thus, the project is a collaboration between DTU Environment and DMI.

## Abstract
DMI and other national and international meteorological institutions use numerical weather prediction (NWP) models to predict precipitation. However, these models tend to misplace or even miss cloudbursts and extreme precipitation events. During the last decades, research on increasing grid resolution of NWP models are investigated as these models might improve the realism of forecasts.
<br>
<br>
This study compares the predictive skill of two deterministic NWP products for extreme rainfall events in Denmark, excluding Bornholm. The two products, an operational NWP model, NEA, and a high-resolution experimental NWP model, DK750, have been compared on 7 hours forecast horizon. Respectively, these NWP products differ by their grid resolution of 2.5km and 0.75km. This near-nowcasting horizon is important for meteorologists when considering making a warning of a possible extreme precipitation event to the public and to agencies who can react.
<br>
<br>
This study aims to investigate the potential of the high-resolution experimental NWP product for short-term predictions of extreme rainfall. This is done by handling, processing, and interpreting NWP data provided by DMI. The processing includes using traditional and neighborhood verification tools to assess forecast differences compared to radar quantitative precipitation estimates (QPE). The interpretation especially involves the NWP products' roles in the operational cloudburst warning system.
<br>
<br>
Results indicate an improved forecast for most extreme rainfall events investigated using DK750. Across fifteen events, DK750 appears to improve the representation of location, peak intensity, and deformation of precipitation with substantial visual improvement for the latter two. The Fractions Skill Score (FSS) improvement is on average 15\% ($\pm$ 13\%) across intensities of 1-20 $\frac{mm}{hr}$ and scales of 2.5-320 km. The largest improvement when using DK750 is for lower scales and intensities. General improvements are also seen when removing bias, where DK750 appears to be relative best for events with a lower wet area ratio, while the reverse is true for NEA. DK750, unlike NEA, tends to underestimate structure and amplitude, which was seen using the object-based verification score structure, amplitude, and location (SAL).
<br>
<br>
There is a potential for using DK750 in the operational cloudburst warning system, where the most considerable potential is seen for smaller and more spread precipitation fields. NEA appears to be more constant in skill across all lead times, suggesting that the largest potential for using DK750 is within a nowcast horizon. Focus on the future should improve spatial skills further and improve forecasts of small and intense precipitation cells.
<br>
<br>

## Content
Ten Python scripts have been used for this study, though this could have been limited. These are divided into whether the basis of them were provided by DMI or not, though these are strongly interconnected.

### Description of the scripts
The first three scripts have been provided by DMI and vary in what has been changed.
<ol>
 <li><b> “identify_cloudbursts_in_stationdata.py”</b> was provided by DMI (Jonas) and is only slightly changed. The script finds times where rain gauge exceeds a chosen threshold. This has only been altered to include production of txt. Files that only contains a string with timestamps to be used later in other scripts. The script produces pdf of an overview over Denmark and which rain gauges exceeded the threshold of choice. This function has not been altered. The last function in the plot, “dates_for_plot”, was made by me to produce a data overview of the days where there was found exceedances, either above a threshold of 15mm/hr or 24 mm/6hr. This plot can also be found in the report.</li>
<br>

 <li><b>“radar_data_handling.py” </b>was also given initially by DMI, but does include a few new functions. It was received with the following functions which remain unchanged:
   <ul><li>	“read_raw_radardata”. Finds where the data is placed in the h5py file. </li>
     <li> “raw_radardata_to_dbz”. Converts the input data to dbz values using a linear relationship of Z=-32+0.5*data. Values equal to 255 is set to NaN. </li>
<li> “dbz_to_R_marshallpalmer”. Calculate estimated rainfall intensities by using constant marshall-palmer relationship of a=200 and b=1.6. </li>
<li> “project_raster_coords”. Transforms gridded data into another user-defined grid. </li>
<li> “remove_values_below”. Small function which sets values below a certain intensity to NaN. Used for plotting in order to distinguish individual precipitation cells.</li>
<li> “radar_plot”. Plotting an image of radar estimated precipitation intensity over a map of Denmark using DMIs colorcoding for intensities. </li>
<li>“Data_to_raster_RADAR” produces a raster of a single hour of radar rainfall estimate image. This has only been changed slightly from the original function given by DMI</li> </ul>

   <i>This python script also contains functions produced by me. This includes the following: </i>
<ul><li> “movefiles”. The radar test-files used for the most part of the project contained both interpolations, nowcasts and a flow file. This functioned eased this mix by moving the files of interest.</li>
<li> “aggregate_data”. This functions refers to several of the given DMI functions to limit repeating the process. This function can then have the given data as input and can return hourly rainfall intensity estimate.</li>
<li> “produce_zonalstat_radar”. Produces zonal statistics by producing a raster image before comparing that to the common grid. </li>
<li> “radar_plot_all_events”. Used to plot the 15 events used in this study. The function is a modification “radar_plot” but allows several subfigures to appear. </li>This plot can be found in the thesis. </li> </ul>

<i> The following three scripts has been used to produce data for all sources in the common 2.5x2.5km grid. These could probably have been done in one, but due to the longevity of making zonal statistics all several months of minute/hourly data, this was done one-by-one. These does not contain any functions, but instead runs functions from the other scripts. </i> <br>

<li> <b>“nwp750_read_plot.py”</b> was initially given by DMI to ease the process of extracting the necessary NWP data. The following scripts remain unchanged:
<ul><li>“read_parameter_info”. Extracts forecast and coordinates values among many other parameters that the model predicts. </li> 
<li>“Output_rain_NWP”. Opens the files given and extracts the necessary data given that it is parameter 58 that is of interest. </li> 
<li>“nwp_plot” Plots the gridded NWP data over a map of data similar to the “radar_plot” function. </li> <br>

  <i>These have been worked out by me with some basis in similar functions:</i>
<li>“data_to_raster_NWP”. Similar to the “Data_to_raster_RADAR” file but with another projection, thereby not using a proj4-string. </li> 
<li>“unaccumulate”. Automatizes opening several files and returning their precipitation values in hourly rainfall intensities. </li> </ul> </li>

<br><i> The remaining script have been produced by me </i><br>

<li> <b>“Main_radarrun.py” </b>
The script is used to aggregate the given radar data into hourly rainfall estimates with a horizontal resolution of 2.5x2.5km. This also means not producing files for days where there is missing data. </li>
<li><b>“Main_NWPrun.py” </b>
Is used to aggregate the given DK750 data into hourly rainfall estimates with a horizontal resolution of 2.5x2.5km. This is done by seeing what NEA data is available, since the need to have the same time of forecast to be useful. </li> <br> <br>
<li> <b>“Main_NEArun.py” </b><br>
Get the data within its’ own grid using zonal statistics like the others to ease the process.  </li>
<li> <b>“plotting_functions.py”</b> <br>
<ul><li>“plot_together”. Plots two different data sources with their respective coordinates in a side-by-side view. </li> <br>
<li>“plot_all”. Used to plot the radar product together with the two NWP products in a side-by-side view. </li>
<li>“plot_3x3”. Plots used for example plots used in this thesis. Takes three different data sources and 3 different lead times. </li></ul><br> <br>
<li> <b>“Other_functions.py” </b><br>
<ul><li> “precipitation_events”. Needs an input of dates and time where a rain gauge had an exceedance of a given threshold. This function compares which exceedance times that overlap with the available time of forecasts the NWP products.  </li>
<li>“produce_zonalstat”. Produces zonal statistics depending on which data type was used as input. This is done as the method for producing a raster I different among the files and their projection is also different meaning different projections of the common grid is also needed.  </li>
<li>“extract_zonalstat”. The zonal statistics form above is formatted in a dictionary, and a non-constant number of gridcells fit inside the common grid meaning that the sum should be divided with the count of cells which is available in this dictionary.  </li>
  <li>“zone-to_2d”. The zonal statistics are reshaped to fit the common grid.  </li>
<li> “produce_gifs”. Produces several pictures using “plot_all” plotting function across multiple dates where matching of radar and NWP products has been made.</li>
<li>“make_gif”. Produces gifs by collecting photos in an input-path into a duration of choice. </li> </ul>
 <li> <b>“Verification.py”</b>
<ul><li>“Fractional_skillscore” is the calculation of FSS depending on if it is an intensity of percentile that has been given. </li>
<li>“Spatial_threshold_matrix”. Inputs a scale of scale and intensity to the function above. This is used for the function below. </li>
<li>“produce_FSS_matrix”. Plots a matrix form series of FSS calculations one an intensity / scale view used in this thesis with a range of dates for different models.</li>
<li>“produce_fss_percentiles” Since the function used to calculate FSS only take a threshold in a rainfall intensity, this function serves as a workaround of finding the quantiles of the radar product and nwp product to multiply the rain intensities above the threshold for both products to ensure same percentile. Examples given in the thesis and the remaining in appendix. </li>
<li>“saveFSS” Saves FSS values in a csv. file</li>
<li>“scalemin_plot”. The plot to find the minimum scale above FSSuniform for a range of date and percentiles in order to find a trend. Found in appendix.</li>
<li>“plot_fss_boxplot”. Produces a boxplox from a range of scales for a given intensity to see the distribution of FSS for the used extreme precipitation events. The output can be found in the thesis. </li>
<li>“plot_fss_leadtime_boxplot”. Produces a boxplot for a given scale and a given intensity divided into lead times to see the variation between the models. The output can be found in the thesis. </li>
<li>“objects”. Is used to plot example how the SAL verification method finds objects with a given threshold of 0.95 and f=1/15. The output, which is 2d-data of indixed precipitation cells is put into the following function. </li>
<li>“plot_objects”. Plot the indixed precipitation cells. The output can be found in the thesis.</li>
<li>“produce_SAL”. Finds structure, amplitude and Location components for a observed and predicted 2d field.</li>
<li>“plot_SAL” Plots the finding from the previous function. The result can be seen in the thesis.</li>
<li>“SAL_output”. Produces and plots SAL across multiple dates as input. </li>
<li>“plot_SAL_l”. Plot the l component together with another component to see that visually.</li></ul>
<li> <b> “Main.py”</b><br>
This script does not contain any functions but is merely from here that testing and using of the functions from the other scripts has happened. Within here, some calculations has also been made for use in the thesis. This also includes some hard-coding in order to produce plots to the report of chosen examples. 

</ol>
### Get started
* Clone the repo
   ```sh
   git clone https://github.com/OliverFjellander/HighresNWP.git
   ```
* Install required modules
  ```sh
  pip install -r requirements.txt
  ```

## Contact
Oliver Hede Fjellander - oliverfjellander@gmail.com

## Acknowledgments
This thesis has fulfilled my personal objective to get involved in a programming-centered project. I want to thank my supervisor Jonas Weid Pedersen, who has guided me daily with everything from discussing proper methods and graphical output to inviting me into a friendly and curious atmosphere at DMI. Thank you to Peter Steen Mikkelsen for his role of both being a first-time listener and questioning ongoing findings and giving valuable perspectives to the project.
For the last half of the project, I had the pleasure of sharing everyday life with employees in the Hydrology and Flooding department at DMI. It was a tremendous relief to be among dedicated and friendly people again. Altogether, I felt like part of a team, and they showed a sincere interest in my project. I would especially like to thank Xiaohua Yang from Research and Development at DMI, who had essential input to my project. Since he assisted in developing the experimental NWP product investigated in this thesis, personal communication with him is a source due to the limited official documentation. 
<br>
Thank you to Kiri Koppelgaard, who has spent several afternoons and evenings reading, commenting, and proofreading parts of this thesis. This altogether made this thesis much more readable.

