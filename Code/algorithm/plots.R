
temp = list.files(pattern="*.csv")
temps = list.files(pattern="sec_nopca.csv")
temp=setdiff(temp,temps)
temp2=gsub(pattern = 'sec.csv',replacement = '',x=temp)
temp22=gsub(pattern = 'sec_nopca.csv',replacement = '',x=temps)

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
myfiless = lapply(temps, read.csv,header=FALSE)
myfiless=do.call(rbind,myfiless)
myfiless$data_size=1
myfiless$algorithm=c('Expectation Maximisation','Relabelling','Importance Reweighting')
myfiless$dataset=c('CIFAR without PCA','CIFAR without PCA','CIFAR without PCA')
myfiless$`Running time`=temp32
data=seq(10000,2000,by=-1000)
for (i in 1:3){
  myfiless$data_size[seq(i,27,by=3)]=data
}


#################
require(ggplot2)
df1=melt(myfiles, id=17:20)
df1=df1[,c(1,2,3,4,6)]
names(df1)=c( "Sample size", "Algorithm", "Data set",  'Average running time (seconds)', "Accuracy"   )
ggplot(df1, aes(x=`Sample size`, y=`Average running time (seconds)`,colour=Algorithm,fill=`Data set`)) +
geom_point(aes(shape=`Data set`),size=3)
#+geom_smooth()+ ylim(0, 39)
ggsave(filename = 'speed.pdf',width = 7, height = 4, units = "in")
ggplot(df1, aes(x=`Sample size`, y=`Accuracy`,colour=Algorithm,fill=`Data set`)) +
geom_point(aes(shape=`Data set`),size=3) +
geom_smooth(alpha=0.5)
ggsave(filename = 'accuracy.pdf',width = 7, height = 4.5, units = "in")
