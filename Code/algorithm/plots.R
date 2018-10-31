setwd("C:/Users/chenc/Documents/GitHub/image_classification/code/algorithm")

temp = list.files(pattern="*.csv")
setwd("C:/Users/chenc/Documents/GitHub/image_classification/code/algorithm/nopca")

temps = list.files(pattern="*.csv")
myfiless = lapply(temps, read.csv,header=FALSE)
myfiless=do.call(rbind,myfiless)
setwd("C:/Users/chenc/Documents/GitHub/image_classification/code/algorithm")
temp=setdiff(temp,temps)
temp=temp[1:54]
temp2=gsub(pattern = 'sec.csv',replacement = '',x=temp)
temp22=gsub(pattern = 'sec.csv',replacement = '',x=temps)

require('stringr')
temp2=str_sub(temp2, start = 20, end = -1L)
temp22=str_sub(temp22, start = 20, end = -1L)
temp3=as.numeric(gsub("[^0-9.]", "", temp2) )/16
temp32=as.numeric(gsub("[^0-9.]", "", temp22) )/16
myfiles = lapply(temp, read.csv,header=FALSE)
myfiles=do.call(rbind,myfiles)
myfiles$data_size=1
myfiles$algorithm=c('Expectation Maximisation','Relabelling','Importance Reweighting','Expectation Maximisation','Relabelling','Importance Reweighting')
myfiles$dataset=c('MNIST','MNIST','MNIST','CIFAR with PCA','CIFAR with PCA','CIFAR with PCA')
myfiles$`Running time`=temp3
data=seq(10000,2000,by=-1000)
for (i in 1:6){
myfiles$data_size[seq(i,54,by=6)]=data
}
#############


myfiless$data_size=1
myfiless$algorithm=c('Expectation Maximisation','Relabelling','Importance Reweighting')
myfiless$dataset='CIFAR without PCA'
myfiless$`Running time`=temp32
data=seq(10000,2000,by=-1000)

for (i in 1:3){
  myfiless$data_size[seq(i,27,by=3)]=data
}
myfiles=rbind(myfiles,myfiless)
#with(myfiless,order(dataset,algorithm))
require('data.table')
library(scales) 
#################
require(ggplot2)
require(dplyr)
df1=melt(myfiles, id=17:20)
df1=df1[,c(1,2,3,4,6)]
lm(data=myfiless[seq(1,27,by=3),],formula=log(`Running time`)~log(data_size))
names(df1)=c( "Sample size", "Algorithm", "Data set",  'Average running time (seconds)', "Accuracy"   )
ggplot(df1, aes(x=`Sample size`, y=`Average running time (seconds)`,linetype=`Data set`,colour=Algorithm,fill=`Data set`)) +
geom_point(aes(shape=`Data set`),size=3)+
  scale_x_log10(
  ) +
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +geom_smooth(method='lm',se=F)
ggsave(filename = 'speed.pdf',width = 7, height = 5, units = "in")

ggplot(df1, aes(x=`Sample size`, y=`Accuracy`,colour=Algorithm,fill=`Data set`,linetype=`Data set`)) +
geom_point(aes(shape=`Data set`),size=3) +
geom_smooth(alpha=0.5)
ggsave(filename = 'accuracy.pdf',width = 7, height = 4.5, units = "in")



tempd=df1%>%filter(`Sample size`==4000)
tempd=tempd[1:6,]
tempn=df1%>%filter(`Sample size`==10000)
tempn=tempn[1:6,]
log(tempn[4]/tempd[4])/log(2.5)#2
tempd=df1%>%filter(`Data set`=='CIFAR with PCA')
tempd=tempd[1:27,]
tempn=df1%>%filter(`Data set`=='MNIST')
tempn=tempn[1:27,]
log(tempn[4]/tempd[4])/log(784/100)
