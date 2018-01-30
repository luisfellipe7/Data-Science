#Loading libraries

library('ggplot2')
library('readr')
library('gridExtra')
library('grid')
library('plyr')
library('GGally')

#Loading dataset
iris=read.csv('.../Iris.csv')

#getting a random sampling of data
iris[sample(nrow(iris),10),]

#First thing we'll do is a density and frequency analysis with a histogram

#Sepal length
SlHist <- ggplot(data=iris, aes(x=SepalLengthCm)) + geom_histogram(binwidth = 0.2, color="black", aes(fill=Species)) +
  xlab("Sepal Length (Cm)") + 
  ylab("Frequency") +
  theme(legend.position = "none")+
  ggtitle("Histogram of Sepal Length")+
  geom_vline(data=iris, aes(xintercept=mean(SepalLengthCm)),linetype="dashed",color="grey")

# Sepal width
SwHist <- ggplot(data=iris, aes(x=SepalWidthCm)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Sepal Width (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")+
  ggtitle("Histogram of Sepal Width")+
  geom_vline(data=iris, aes(xintercept = mean(SepalWidthCm)),linetype="dashed",color="grey")


# Petal length
PlHist <- ggplot(data=iris, aes(x=PetalLengthCm))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Petal Length (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="none")+
  ggtitle("Histogram of Petal Length")+
  geom_vline(data=iris, aes(xintercept = mean(PetalLengthCm)),
             linetype="dashed",color="grey")



# Petal width
PwHist <- ggplot(data=iris, aes(x=PetalWidthCm))+
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Petal Width (cm)") +  
  ylab("Frequency") + 
  theme(legend.position="right" )+
  ggtitle("Histogram of Petal Width")+
  geom_vline(data=iris, aes(xintercept = mean(PetalWidthCm)),linetype="dashed",color="grey")


# Plot all visualizations
grid.arrange(SlHist + ggtitle(""),
             SwHist + ggtitle(""),
             PlHist + ggtitle(""),
             PwHist  + ggtitle(""),
             nrow = 2,
             top = textGrob("Iris Frequency Histogram", 
                            gp=gpar(fontsize=15))
)

# We can review the density distribution of each attribute broken down by class value. 
# Like the scatterplot matrix, the density plot by class can help see the separation of classes. 
# It can also help to understand the overlap in class values for an attribute.

#Density Petal Length
PlDhist <- ggplot(iris, aes(x=PetalLengthCm, colour=Species, fill=Species))+
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(PetalLengthCm), colour=Species),linetype="dashed",color="grey", size=1)+
  xlab("Petal Length (cm")+
  ylab("Density")+
  theme(legend.position = "none")

#Petal Width
PwDhist <- ggplot(iris, aes(x=PetalWidthCm, colour=Species, fill=Species))+
  geom_density(alpha=.3)+
  geom_vline(aes(xintercept=mean(PetalWidthCm),  colour=Species),linetype="dashed",color="grey", size=1)+
  xlab("Petal Width (cm)") +  
  ylab("Density")

#Sepal Width
SwDhist <- ggplot(iris, aes(x=SepalWidthCm, colour=Species, fill=Species)) +
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(SepalWidthCm),  colour=Species), linetype="dashed",color="grey", size=1)+
  xlab("Sepal Width (cm)") +  
  ylab("Density")+
  theme(legend.position="none")

#Sepal Length
SlDhist <- ggplot(iris, aes(x=SepalLengthCm, colour=Species, fill=Species)) +
  geom_density(alpha=.3) +
  geom_vline(aes(xintercept=mean(SepalLengthCm),  colour=Species),linetype="dashed", color="grey", size=1)+
  xlab("Sepal Length (cm)") +  
  ylab("Density")+
  theme(legend.position="none")

# Plot all density visualizations
grid.arrange(SlDhist + ggtitle(""),
             SwDhist  + ggtitle(""),
             PlDhist + ggtitle(""),
             PwDhist  + ggtitle(""),
             nrow = 2,
             top = textGrob("Iris Density Plot", 
                            gp=gpar(fontsize=15))
)

#Now we need to identify some outliers, as some classes do not overlap
#and some other attributes are hard to terase apart

ggplot(iris, aes(Species, PetalLengthCm, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Petal Length (cm)", breaks= seq(0,30, by=.5))+
  labs(title = "Iris Petal Length Box Plot", x = "Species")

#Now we plt all the variables in a single visualization that will contain all the boxplots

SlBp <- ggplot(iris, aes(Species, SepalLengthCm, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Sepal Length (cm)", breaks= seq(0,30, by=.5))+
  theme(legend.position="none")



SwBp <-  ggplot(iris, aes(Species, SepalWidthCm, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Sepal Width (cm)", breaks= seq(0,30, by=.5))+
  theme(legend.position="none")



PlBp <- ggplot(iris, aes(Species, PetalLengthCm, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Petal Length (cm)", breaks= seq(0,30, by=.5))+
  theme(legend.position="none")



PwBp <-  ggplot(iris, aes(Species, PetalWidthCm, fill=Species)) + 
  geom_boxplot()+
  scale_y_continuous("Petal Width (cm)", breaks= seq(0,30, by=.5))+
  labs(title = "Iris Box Plot", x = "Species")



# Plot all visualizations
grid.arrange(SlBp  + ggtitle(""),
             SwBp  + ggtitle(""),
             PlBp + ggtitle(""),
             PwBp + ggtitle(""),
             nrow = 2,
             top = textGrob("Sepal and Petal Box Plot", 
                            gp=gpar(fontsize=15))
)

#Using violin plots
#Violing plots are similar to box plot but show the number of point at a
#particular value by the width of the shapes
#also it can contain the marker for the median and a box for the interquartile range

SlVp <-  ggplot(iris, aes(Species, SepalLengthCm, fill=Species)) + 
  geom_violin(aes(color = Species), trim = T)+
  scale_y_continuous("Sepal Length", breaks= seq(0,30, by=.5))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")

SwVp <-  ggplot(iris, aes(Species, SepalWidthCm, fill=Species)) + 
  geom_violin(aes(color = Species), trim = T)+
  scale_y_continuous("Sepal Width", breaks= seq(0,30, by=.5))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")



PlVp <-  ggplot(iris, aes(Species, PetalLengthCm, fill=Species)) + 
  geom_violin(aes(color = Species), trim = T)+
  scale_y_continuous("Petal Length", breaks= seq(0,30, by=.5))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")




PwVp <-  ggplot(iris, aes(Species, PetalWidthCm, fill=Species)) + 
  geom_violin(aes(color = Species), trim = T)+
  scale_y_continuous("Petal Width", breaks= seq(0,30, by=.5))+
  geom_boxplot(width=0.1)+
  labs(title = "Iris Box Plot", x = "Species")


# Plot all visualizations
grid.arrange(SlVp  + ggtitle(""),
             SwVp  + ggtitle(""),
             PlVp + ggtitle(""),
             PwVp + ggtitle(""),
             nrow = 2,
             top = textGrob("Sepal and Petal Violin Plot", 
                            gp=gpar(fontsize=15))
)

#Next we need to create a scatterplot of petal lengths vs petal width
#with the color and shape by species
#There is also a regression line with 95% confidence band

ggplot(data = iris, aes(x = PetalLengthCm, y = PetalWidthCm))+
  xlab("Petal Length")+
  ylab("Petal Width") +
  geom_point(aes(color = Species,shape=Species))+
  geom_smooth(method='lm')+
  ggtitle("Petal Length vs Width")


# Here is a similar plot with more details on the regression line.
library(car)
scatterplot(iris$PetalLengthCm,iris$PetalWidthCm)

#Checking the Sepal Length vs Width
ggplot(data=iris, aes(x = SepalLengthCm, y = SepalWidthCm)) +
  geom_point(aes(color=Species, shape=Species)) +
  xlab("Sepal Length") + 
  ylab("Sepal Width") +
  ggtitle("Sepal Length vs Width")

# Based on all the plots we have done we can see there is certain correlation. Let's take a look at the pairwise correlation numerical values to 
# ascertain the relationships in more detail.

library(GGally)
ggpairs(data = iris[1:4],
        title = "Iris Correlation Plot",
        upper = list(continuous = wrap("cor", size = 5)), 
        lower = list(continuous = "smooth")
)

# Examining the plot reveals atrong correlation between the variables 
#Petal Width and Length
#as well as Sepal Lenght and Petal Lenght
#We also can use a heatmap as exploratory plot
#It functions as a 2D Histogram
#The brighter the color the larger the value

irisMatix <- as.matrix(iris[1:150, 1:4])
irisTransposedMatrix <- t(irisMatix)[,nrow(irisMatix):1]

image(1:4, 1:150, irisTransposedMatrix)