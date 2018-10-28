# First competition on kaggle 
top 3% solution for 2018 dsb.
Althrough it is 6 month ago, but it seems good to record every competition on github.
It was the first thing to find a proper competition on kaggle When I finished deeplearning courses from Andrew Ng,
and 2018 data science bowl was what I found.
The task of this competition is to segment the nucleus of cells in various kinds of images.

## Start from public kernel
I didn't have any experience at that time, so I just grabbed a public kernel to begin with a simple UNet.
After some struggling with keras unet, I got a very bad result.
But since it is the first time, nothing to fear, I kept trying.

## No human labor, no intelligence
I found some pattern in the training images after some thinking.
The training images can be divided into 5 main catergories:
1. Black-White images with massive small cells.
2. Black-White images with massive cells in clusters.
3. Black-White images with rectangle cells and clear nucleus in it.
4. Pink-White images with massive small cells.
5. Pink-White images with huge cell and clear nucleus in it.

So the idea to train a image classifier before unet come into my head.
I just mannualy split the train images into 5 catergories and trained the classifier with VGG style network.
It was lucky that there are 670 images and I finished my mannual classification in 4 hours.
(To be honest, I will never do such stupid thing again.)
Then I trained my classifier, after that I trained 5 unet with each type of images.
I got a slightly increase in my LB score but nothing exiting.
The reason for the poor result is that the No.3 catergory did not present in the stage 1 test data.

## Step into MRCNN
After about 1 week struggle in unets, I read a post in the form forum which says mask-rcnn did quite well
in this competition. So I headed to google and found the keras implementation of mrcnn by matterport.
It was lucky again I was a programmer to understand the codes from the github. 
But shortly I found it was
not that great to this competition:
1. The train process is too slow and take 2 days to finish a round.
2. The result messed up with all kinds shapes of nucleus. 
  Couldn't mrcnn just learn the nucleus should be an oval?
Then I realized that that's because I trained it without pretrained weights and the data is really too small
for mrcnn to learn.
So I tried many argumentation methods to increase the amount of training data from 670 to 4000.
And that makes me sleepless because it took more than 5 days to finish a round.
My programmer experience saved me again to load all the data into main memory and add earlystop and learning rate
scheduler in training.
So I reduced the training time to 20 hours.
And I can train 5 mrcnns after my classifier.
Again, this is a stupid idea to waste time but it really helped a little with my LB score.

## From The Grand Masters
But I stopped improving when I finally get to the top 9%.
There was really no more ideas from myself.
Then the grand master Heng CherKeng published a serials of post talking about his own mrcnn implementation in pytorch.
I believe he was, is and will be the true master in kaggle community who really understand the details of mrcnn.
![Heng CherKeng's mrcnn](https://www.researchgate.net/profile/Yan_Xu4/publication/305401350/figure/fig2/AS:385845317128192@1469004100993/This-illustrates-the-structure-of-this-framework-We-fuse-outputs-of-three-channels-to.ppm)

Unfortunately, I didn't learnt pytorch at that moment.
I can only try to modify the matterport code into his idea.
Althrough I didn't succeed in this, but I figured out why my score stopped improving.
One reason for that is I used only 128 ROI in each images. 
But some images contains more than 200 cells and nucleus in them.
The other reason is the inappropriate RPN_ANCHOR_SCALES which can not handle both small cells and huge cells.

So, I changed my manual classify strategy and re-select the cadidates for the sub classes.
That's all I've done in this competition.

## Conclution
Heng CherKeng didn't win the first place in this competition, but his idea inspired a lot of people to 
get a very good position.
It is a surprise that the winner Victor Durnov and his team was using UNet with softmax in the end to predict
both the nucleus and the border of cells. It turned out the best solutions to handle the massive cells.

![1st place idea](https://www.dropbox.com/s/4igam47pqg0i82q/c43e356beedae15fec60ae3f8b06ea8e9036081951deb7e44f481b15b3acfc37_predict.png)

After reading their github repository, I found it was not all.
They trained a lightgbm after unet found the segmentations to check if each result is really a nucleus.


I really have too much to learn, and I love the way kaggle spread ideas.

