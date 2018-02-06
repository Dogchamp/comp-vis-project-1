## Computer Vision Project 1
This is a write up for the Computer Visions course at GWU.

### The Algorithm
This colorizer algorithm will first divy up the image into each of the color channels by dividing the image into horizontal thirds, and assigning each their correct color (RGB from top-to-bottom) 
As an extra feature, it will then correct the contrast of the entire picture via CLAHE (Contrast Limited Adaptive Histogram Equalization). openCV's CLAHE module preferse operating on grayscale pictures it seems, so this part is done pre-alignment.
Next, the alignment process begins. It does so by first constructing the Gaussian image pyramid down to four levels for each channel. The Gaussian image pyramid is contstructed by applying a Gaussian filter to the image and then sampling every other pixel from the image to create a new one.
Then, each level from the pyramid is scored against its corresponding level in the next channel. For example, red will be scored against green, and green to blue. Scoring is done via a sum of squared differences (ssd) operation. The displacement is found for the best score (the lowest ssd), and then the image is then rolled to align.
The images are then stacked ontop of each other to form the final image. 	

#### Clahe
Here is CLAHE in action. The difference is very visible.
![Image](gramma.jpg)
![Image](gramma_clahe.jpg)

