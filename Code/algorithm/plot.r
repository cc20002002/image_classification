temp = list.files(pattern="*.csv")

myfiles = lapply(temp, read.csv,header=FALSE)

myfiles=do.call(rbind,myfiles)
myfiles=myfiles[-1,]
myfiles2=myfiles[4:6,]
#myfiles=t(myfiles)
require('data.table')
myfiles2$algorithm=c('Expectation Maximisation with PCA','Relabelling with PCA','Importance Reweighting with PCA')
myfiles2$dataset=c('CIFAR','CIFAR','CIFAR')
df1=melt(myfiles2, id=17:18)

df1[seq(1,48,by=3),5]=round(35.5/16)
df1[seq(2,48,by=3),5]=round(195.5/16)
df1[seq(3,48,by=3),5]=round(222.2/16)

df1[,5]=as.factor(df1[,5])
names(df1)=c("Algorithm" ,"Dataset",   "Average running time (seconds)",  "Accuracy", "Average running time (seconds)" )
df2=df1[,c(1,2,4,5)]
require('ggplot2')
library(ggthemes)
df2$`Average running time (seconds)`=as.numeric(as.character(df2$`Average running time (seconds)`))
df22=df2
save(list='df22',file = 'pca')
theme_set(theme_bw())  # from ggthemes
ggplot(df2,aes(x=`Average running time (seconds)`,y=Accuracy,color=Algorithm,fill=Dataset))+
                 geom_boxplot(size = 1) + scale_fill_hue(l=100, c=100,h.start=330)+
  coord_flip()+ theme(legend.position="top")+
  guides(fill=guide_legend(ncol=1,nrow=3,byrow=TRUE),color=guide_legend(ncol=1,nrow=3,byrow=TRUE))+
  scale_x_continuous(breaks = pretty(df2$`Average running time (seconds)`, n = 10)) +
  scale_y_continuous(breaks = pretty(df2$Accuracy, n = 10))
ggsave(filename = 'boxplot.pdf',width = 7, height = 7, units = "in")
ggplot(df2,aes(x=`Average running time (seconds)`,y=Accuracy,color=Algorithm,fill=Dataset))+
  geom_boxplot(size = 1) + scale_fill_hue(l=100, c=100,h.start=330)+
  scale_x_continuous(breaks = pretty(df2$`Average running time (seconds)`, n = 10)) +
  scale_y_continuous(breaks = pretty(df2$Accuracy, n = 10))
ggsave(filename = 'boxplotv.pdf',width = 7, height = 7, units = "in")
Datasets=unique(df2$Dataset)
plots=list()
require('dplyr')
for (nn in Datasets){
  g=ggplot(df2 %>% filter(Dataset==nn), aes(x=Accuracy, fill=Algorithm)) +
    geom_density(alpha=0.5, position="identity")+
    ylab("Probability density")
  plots=c(plots,list(g+theme(legend.position="none")))
  
}
require('cowplot')
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
ks.test(myfiles[1,1:16],t(myfiles[2,1:16]))# em vs relabeling 
ks.test(myfiles[1,1:16],t(myfiles[3,1:16]))# em vs reweighting

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
