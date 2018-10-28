set.seed(1234)
library(e1071)
probs <- cbind(c(.4,.2/3,.2/3,.2/3,.4),c(.1/4,.1/4,.9,.1/4,.1/4),c(.2,.2,.2,.2,.2))
my.n <- 100
my.len <- ncol(probs)*my.n
raw <- matrix(NA,nrow=my.len,ncol=2)
raw <- NULL
for(i in 1:ncol(probs)){
  raw <- rbind(raw, cbind(i,rdiscrete(my.n,probs=probs[,i],values=1:5)))
}
raw <- data.frame(raw)
names(raw) <- c("group","value")
raw$group <- as.factor(raw$group)
raw.1.2 <- subset(raw, raw$group %in% c(1,2))
x=data.frame(group=rep(1,100),likert=rep(1,100)
           )
x$group[51:100]=2
x$likert=as.numeric(x$likert)
x$likert[1:22]=6
x$likert[23:41]=5
x$likert[42:50]=4
x$likert[51:85]=6
x$likert[86:97]=5
x$likert[98:100]=4
x$likert=factor(x$likert,ordered = T)
x$group=factor(x$group,ordered = F)
wilcox.test(likert ~ group, 
            data=x)
require(MASS)
a=polr(likert ~ group, 
     data=x)
pr <- profile(a)
confint(pr)
coeffs <- coef(summary(a))
p <- pnorm(abs(coeffs[, "t value"]), lower.tail = FALSE) * 2
cbind(coeffs, "p value" = round(p,3))