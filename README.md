
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

### Abstract
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

### Get started
* Clone the repo
   ```sh
   git clone https://github.com/OliverFjellander/HighresNWP
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

