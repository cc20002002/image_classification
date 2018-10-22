temp = list.files(pattern="*.csv")
myfiles = lapply(temp, read.csv,header=FALSE)
myfiles=do.call(rbind,myfiles)
#myfiles=t(myfiles)
require('data.table')
myfiles$algorithm=c('Expectation Maximisation','Relabelling','Reweighting','Expectation Maximisation','Relabelling','Reweighting')
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
library(brms)
theme_set(theme_bw())  # from ggthemes
ggplot(df2,aes(x=`Average running time (seconds)`,y=Accuracy,color=Algorithm,fill=Dataset))+
                 geom_boxplot(size = 1) + scale_fill_hue(l=100, c=100)
ggsave(filename = 'boxplot.pdf',width = 7, height = 7, units = "in")
  
Datasets=unique(df2$Dataset)
plots=list()
for (nn in Datasets){
  g=ggplot(df2 %>% filter(Dataset==nn), aes(x=Accuracy, fill=Algorithm)) +
    geom_density(alpha=0.5, position="identity")
  plots=c(plots,list(g+theme(legend.position="none")))
  
}
legend <- get_legend(g+theme(legend.position="top"))
plot_grid(plotlist = plots, labels=Datasets,hjust =c(-1,-1)) +
  theme(plot.margin=unit(c(1,0,0,0),"cm"))+
  draw_grob(legend, .45, .53, .3/3.3, 1)+
  ylab("Probability density")
ggsave(filename = 'histo.pdf',width = 7, height = 7, units = "in")

#reweighting is better than em is better than relabelling
ks.test(myfiles[5,1:16],t(myfiles[4,1:16]),alternative ='greater')# em vs relabeling
ks.test(myfiles[4,1:16],t(myfiles[6,1:16]),alternative ='greater')# em vs reweighting
ks.test(myfiles[5,1:16],t(myfiles[6,1:16]),alternative ='greater')# labeling vs reweighting


#relabelling is more accuracy than em
ks.test(myfiles[1,1:16],t(myfiles[2,1:16]),alternative ='greater')# em vs reweighting 
ks.test(myfiles[1,1:16],t(myfiles[3,1:16]))# em vs relabeling

ks.test(myfiles[3,1:16],t(myfiles[2,1:16]))# labeling vs reweighting


  df11[Noise==nn], aes(x=`Relative residual error`, fill=Algorithm)) +
  geom_histogram(alpha=0.2, position="identity",bins=60)+
  ylab("Count")