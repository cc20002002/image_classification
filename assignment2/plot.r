temp = list.files(pattern="*.csv")
temp=temp[-1]
myfiles = lapply(temp, read.csv,header=FALSE)

myfiles=do.call(rbind,myfiles)
#myfiles=t(myfiles)
require('data.table')
myfiles$algorithm=c('Expectation Maximisation','Sample filtering','Importance Reweighting','Expectation Maximisation','Importance Reweighting','Importance Reweighting')
myfiles$dataset=c('MNIST','MNIST','MNIST','CIFAR','CIFAR','CIFAR')
df1=melt(myfiles, id=17:18)

df1[seq(1,96,by=6),5]=round(209.2031/16)
df1[seq(2,96,by=6),5]=round(942.7264/16)
df1[seq(3,96,by=6),5]=round(1059.5852/16)
df1[seq(4,96,by=6),5]=round(812.642/16)
df1[seq(5,96,by=6),5]=round(3742.079/16)
df1[seq(6,96,by=6),5]=round(4208.5504/16)
df1[,5]=as.factor(df1[,5])
names(df1)=c("Algorithm" ,"Dataset",   "Average running time (seconds)",  "Accuracy", "Average running time (seconds)" )
df2=df1[,c(1,2,4,5)]
require('ggplot2')
library(ggthemes)

theme_set(theme_bw())  # from ggthemes
ggplot(df2,aes(x=`Average running time (seconds)`,y=Accuracy,color=Algorithm,fill=Dataset))+
                 geom_boxplot(size = 1) + scale_fill_hue(l=100, c=100)
ggsave(filename = 'boxplot.pdf',width = 7, height = 5, units = "in")
  
Datasets=unique(df2$Dataset)
plots=list()
for (nn in Datasets){
  g=ggplot(df2 %>% filter(Dataset==nn), aes(x=Accuracy, fill=Algorithm)) +
    geom_density(alpha=0.5, position="identity")+
    ylab("Probability density")
  plots=c(plots,list(g+theme(legend.position="none")))
  
}
require(cowplot)
legend <- get_legend(g+theme(legend.position="top"))
plot_grid(plotlist = plots, labels=Datasets,hjust =c(-1,-1),vjust=c(2,2)) +
  theme(plot.margin=unit(c(1,0,0,0),"cm"))+
  draw_grob(legend, .45, .53, .3/3.3, 1)
ggsave(filename = 'histo.pdf',width = 7, height = 4.6, units = "in")

#reweighting is better than em is better than relabelling
ks.test(myfiles[5,1:16],t(myfiles[4,1:16]))# em vs relabeling
ks.test(myfiles[4,1:16],t(myfiles[6,1:16]))# em vs reweighting
ks.test(myfiles[5,1:16],t(myfiles[6,1:16]))# labeling vs reweighting


#relabelling is more accuracy than em
ks.test(myfiles[1,1:16],t(myfiles[2,1:16]))# em vs reweighting 
ks.test(myfiles[1,1:16],t(myfiles[3,1:16]))# em vs relabeling

ks.test(myfiles[3,1:16],t(myfiles[2,1:16]))# labeling vs reweighting
require('boot')
df3=myfiles
df3$Mean=rowMeans((df3[,1:16]))
df3$Sd=apply(df3[,1:16],1,sd)
mean.fun <- function(dat, idx) mean(dat[idx], na.rm = TRUE)
for (i in 1:6){
bootobject <- boot(data=df1[seq(i,96,by=6),4],R=1000,statistic=mean.fun)
a=boot.ci(bootobject, type='perc' )
df3[i,]$CIl=a$percent[4]
  df3[i,]$CIu=a$percent[5]
}
df4=round(df3,3)
fwrite(df4,file = 'meansd.csv')
