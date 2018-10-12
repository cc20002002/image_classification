require(data.table)
DT <- data.table(
  Y = rbinom(50,1,0.5),
  S = 1
)
DT$Y=1
n=nrow(DT)
indy1=sample(1:n,size = .5*n,replace = F)
DT$Y[indy1]=0
DT$S=DT$Y
index0=which(DT$Y==0)
index0=sample(index0,size = length(index0)*0.2)
index1=which(DT$Y==1)
index1=sample(index1,size = length(index1)*0.4)
DT$S[index0]=1-DT$S[index0]
DT$S[index1]=1-DT$S[index1]
p=0.2
s=c()
DT=data.frame(DT)
for (i in 3:8){
  DT=cbind(DT,DT[,ncol(DT)])
  names(DT)[ncol(DT)]='S'
  DT[,i]=DT[,i-1]
  index0=which(DT[,i-1]==0)
  index0=sample(index0,size = length(index0)*0.25*p)
  index1=which(DT[,i-1]==1)
  index1=sample(index1,size = length(index1)*0.33*p)
  DT[index0,c(i)]=1-DT[index0,c(i)]
  DT[index1,c(i)]=1-DT[index1,c(i)]
  s=c(s,sum(DT[,c(i)]))
}