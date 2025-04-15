# COMPUTER VISION CAPABILITIES FOR AQUATIC NAVIGATION
> This repository contains all the scripts done by the software team of Iaac's engineering club realted to computer vision for boat manouvering.

## Software team:
- [Salvador Canturias](https://www.linkedin.com/in/salvador-cantuarias-bb5715268/)
- [Nouhaila Elmalouli](https://www.linkedin.com/in/nouhaila-elmalouli-46517a208/)

## Tasks:
> **2025-04-11:** Create a class that uses computer vision's color recognition to detect elements in the video feed and make the boat respond accordingly.
>> **status:** DONE
>> **observations:**<br>- The script is able to take pre recorded video inputs. now its time to test live feed.<br>- We need to change the mathematical approach to identifing the postion of the buoys. Right now it is just related to which forizontal half of the frame is being used. this ould be changed by measuring the delta space between the buoys, if said delta space has a postive or negative value and how close is the center of that delta space to the center of the frame for color correction.<br>- More cases need to be tested: when theres more than two buoys, when a buoy is partially obstrucuted, etc...<br>- Cardinal buoys need to be integrated into the class.

> **2025-04-22:**